#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    rustdoc::missing_crate_level_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::{stderr, Write};
use std::ops::DerefMut;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};

use tracing::{span, Event as TracingEvent, Id, Span, Subscriber};
use tracing_subscriber::field::RecordFields;

use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};

const NESTED_EVENT_OFFSET: usize = 2;
const DURATION_WIDTH: usize = 6;

#[derive(Debug, Clone)]
struct EventInfo {
    timestamp: SystemTime,
    metadata: TrackedMetadata,
}

impl EventInfo {
    fn to_string(&self, settings: &Settings) -> String {
        let mut out = String::new();
        self.metadata.write(&mut out, settings).unwrap();
        out
    }
}

impl SpanInfo {
    fn for_span<S>(span: &span::Id, ctx: &Context<'_, S>) -> Self
    where
        S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
    {
        Self {
            name: ctx.metadata(span).unwrap().name().to_string(),
            start: SystemTime::now(),
            end: None,
        }
    }
}

lazy_static::lazy_static! {
    static ref DUMPER: TeXRayLayer = TeXRayLayer::_new();
}

/// Examine a given span with custom settings
///
/// _Note_: A [`TeXRayLayer`] must be installed as a subscriber.
///
/// # Examples
/// ```no_run
/// use tracing_texray::{Settings, TeXRayLayer};
/// use tracing_subscriber::Registry;
/// use tracing::info_span;
/// use tracing_subscriber::layer::SubscriberExt;
/// let layer = TeXRayLayer::new().enable_events();
/// let subscriber = Registry::default().with(layer);
/// tracing::subscriber::set_global_default(subscriber).unwrap();
/// tracing_texray::examine_with(info_span!("hello"), Settings::auto().enable_events()).in_scope(|| {
///   println!("I'm in this span!");
/// });
/// ```
pub fn examine_with(span: Span, local_settings: Settings) -> Span {
    DUMPER.dump_on_exit(&span, Some(local_settings));
    span
}

/// Examine a given span with settings from the base `TeXRayLayer`
///
/// _Note_: A [`TeXRayLayer`] must be installed as a subscriber.
///
/// # Examples
/// ```no_run
/// use tracing_texray::{Settings, TeXRayLayer};
/// use tracing_subscriber::Registry;
/// use tracing::info_span;
/// use tracing_subscriber::layer::SubscriberExt;
/// let layer = TeXRayLayer::new().enable_events();
/// let subscriber = Registry::default().with(layer);
/// tracing::subscriber::set_global_default(subscriber).unwrap();
/// tracing_texray::examine(info_span!("hello")).in_scope(|| {
///   println!("I'm in this span!");
/// });
/// ```
pub fn examine(span: Span) -> Span {
    DUMPER.dump_on_exit(&span, None);
    span
}

#[derive(Default, Debug, Clone)]
struct TrackedMetadata {
    data: Vec<(String, String)>,
}

impl TrackedMetadata {
    fn write(&self, f: &mut impl std::fmt::Write, settings: &Settings) -> std::fmt::Result {
        let relevant_fields = || {
            self.data
                .iter()
                .filter(|(f, _)| settings.field_printing.should_print(f.as_str()))
        };

        if let Some((_, message)) = relevant_fields().find(|(k, _)| k == "message") {
            write!(f, "{}", message.lines().next().unwrap_or_default())?;
        }

        let relevant_fields = || relevant_fields().filter(|(k, _v)| k != "message");

        if relevant_fields().count() == 0 {
            return Ok(());
        }

        write!(f, "{{")?;
        for (k, v) in relevant_fields().take(self.data.len() - 1) {
            write!(f, "{}: {}, ", k, v)?;
        }
        let (k, v) = relevant_fields().last().expect("non empty");
        write!(f, "{}: {}", k, v)?;
        write!(f, "}}")
    }
}

#[derive(Debug, Clone)]
struct SpanInfo {
    start: SystemTime,
    end: Option<SystemTime>,
    name: String,
}

impl SpanInfo {
    fn full_name(&self, tracker: &SpanTracker, settings: &Settings) -> String {
        let mut id = self.name.to_string();
        tracker
            .metadata
            .write(&mut id, settings)
            .expect("rendering to string cannot fail");
        id
    }

    fn render(
        &self,
        out: &mut dyn Write,
        tracker: &SpanTracker,
        settings: &Settings,
        render_conf: &RenderConf,
        left_offset: usize,
    ) -> std::io::Result<()> {
        let mut key = self.full_name(tracker, settings);
        let truncated_key_width = render_conf.key_width - left_offset;
        key.truncate(truncated_key_width);
        let ev_start_ts = self.start;
        let ev_end_ts = match self.end {
            None => return Ok(()),
            Some(ts) => ts,
        };
        let span_len = ev_end_ts.duration_since(ev_start_ts).unwrap();
        if let Some(min_duration) = settings.min_duration.as_ref() {
            if &span_len < min_duration {
                return Ok(());
            }
        }
        if left_offset > 0 {
            write!(out, "{}", " ".repeat(left_offset))?;
        }
        write!(out, "{:width$}", key, width = truncated_key_width)?;
        write!(
            out,
            " {:>dur_width$} ",
            pretty_duration(span_len),
            dur_width = DURATION_WIDTH
        )?;

        let offset = width(
            render_conf.chart_width(),
            render_conf.total(),
            ev_start_ts.duration_since(render_conf.start_ts).unwrap(),
        );
        write!(out, "{}", " ".repeat(offset)).unwrap();
        let interval_width = width(render_conf.chart_width(), render_conf.total(), span_len);
        match interval_width {
            0 => write!(out, "┆"),
            1 => write!(out, "│"),
            2 => write!(out, "├┤"),
            _more => write!(out, "├{}┤", "─".repeat(interval_width - 2)),
        }?;
        writeln!(out)?;
        Ok(())
    }
}

#[derive(Debug)]
struct SpanTracker {
    span_id: span::Id,
    info: Option<SpanInfo>,
    metadata: TrackedMetadata,
    events: Vec<EventInfo>,
    children: HashMap<span::Id, SpanTracker>,
    settings: Option<Settings>,
}

impl Visit for TrackedMetadata {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        self.data.push((field.to_string(), format!("{:?}", value)));
    }
}

fn width(chars: usize, outer: Duration, inner: Duration) -> usize {
    if inner.as_nanos() == 0 || outer.as_nanos() == 0 {
        return 0;
    }
    let ratio = inner.as_secs_f64() / outer.as_secs_f64();
    (ratio * chars as f64).round() as usize
}

#[derive(Debug)]
struct RenderConf {
    start_ts: SystemTime,
    end_ts: SystemTime,
    key_width: usize,
    width: usize,
}

impl RenderConf {
    fn total(&self) -> Duration {
        self.end_ts.duration_since(self.start_ts).unwrap()
    }

    fn chart_width(&self) -> usize {
        self.width
            .checked_sub(self.key_width)
            .and_then(|w| w.checked_sub(DURATION_WIDTH + 2))
            .unwrap_or(20)
    }
}

impl SpanTracker {
    fn new(id: span::Id, settings: Option<Settings>) -> Self {
        Self {
            span_id: id,
            info: None,
            events: vec![],
            metadata: Default::default(),
            children: Default::default(),
            settings,
        }
    }

    fn record_metadata(
        &mut self,
        span: &span::Id,
        path: impl Iterator<Item = span::Id>,
        values: &dyn RecordFields,
    ) {
        self._with_tracker(span, path, |tracker| {
            values.record(&mut tracker.metadata);
        })
    }

    fn add_event(
        &mut self,
        span: &span::Id,
        path: impl Iterator<Item = span::Id>,
        event: EventInfo,
    ) {
        self._with_tracker(span, path, |tracker| tracker.events.push(event));
    }

    fn open(&mut self, span: &span::Id, path: impl Iterator<Item = span::Id>, span_info: SpanInfo) {
        self._with_tracker(span, path, |tracker| match &mut tracker.info {
            Some(_info) => tracing::error!("opening a span twice"),
            None => tracker.info = Some(span_info),
        });
    }

    // TODO: don't need meta here, just id
    fn exit(
        &mut self,
        span: &span::Id,
        path: impl Iterator<Item = span::Id>,
        timestamp: SystemTime,
    ) {
        self._with_tracker(span, path, |tracker| match &mut tracker.info {
            Some(info) => info.end = Some(timestamp),
            None => tracing::error!("Attempting to exit a span that has never been entered"),
        })
    }

    fn path<'a, S>(
        &self,
        span: &span::Id,
        ctx: &'a Context<'a, S>,
    ) -> impl Iterator<Item = span::Id> + 'a
    where
        S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
    {
        let root = self.span_id.clone();
        let mut path = ctx
            .span_scope(span)
            .unwrap()
            .from_root()
            .skip_while(move |span| span.id() != root)
            .map(|s| s.id());
        let path_root = path.next().expect("one item exists");
        assert_eq!(&path_root, &self.span_id);
        path
    }

    fn events(&self) -> impl Iterator<Item = &SpanInfo> {
        self.info.iter().chain(
            self.children.iter().flat_map(|(_, child)| {
                Box::new(child.events()) as Box<dyn Iterator<Item = &SpanInfo>>
            }),
        )
    }

    fn max_key_width(&self, settings: &Settings) -> usize {
        let longest_self = self
            .info
            .as_ref()
            .map(|info| info.full_name(self, settings).len())
            .unwrap_or_default();
        let longest_child = self
            .children
            .iter()
            .map(|(_, child)| child.max_key_width(settings) + NESTED_EVENT_OFFSET)
            .max()
            .unwrap_or(0);
        longest_self.max(longest_child)
    }

    fn dump(&self, settings: &Settings) -> std::io::Result<()> {
        let settings = self.settings.as_ref().unwrap_or(settings);
        let mut out = settings.out.inner.lock().unwrap();
        self.dump_to(out.deref_mut(), settings)
    }

    fn dump_to(&self, w: &mut dyn Write, settings: &Settings) -> std::io::Result<()> {
        let all_events = self.events().collect::<Vec<_>>();
        if all_events.is_empty() {
            return Ok(());
        }
        let (start_ts, end_ts) = (
            all_events
                .iter()
                .map(|ev| ev.start)
                .min()
                .expect("non empty"),
            all_events
                .iter()
                .flat_map(|ev| ev.end)
                .max()
                .expect("non empty"),
        );
        let conf = RenderConf {
            start_ts,
            end_ts,
            key_width: self.max_key_width(settings).min(120),
            width: settings.width,
        };
        self._dump(w, &conf, settings, 0)
    }

    fn _dump(
        &self,
        out: &mut dyn Write,
        render_conf: &RenderConf,
        settings: &Settings,
        left_offset: usize,
    ) -> std::io::Result<()> {
        let span_info = match &self.info {
            Some(info) => info,
            /* span never got data */
            None => return Ok(()),
        };

        if settings.types.spans {
            span_info.render(out, self, settings, render_conf, left_offset)?;
        }

        if settings.types.events {
            let left_offset = left_offset + 2;
            let truncated_key_width = render_conf.key_width - left_offset;
            let base_offset = width(
                render_conf.chart_width(),
                render_conf.total(),
                span_info
                    .start
                    .duration_since(render_conf.start_ts)
                    .unwrap(),
            );
            let mut settings_with_message = settings.clone();
            if let FieldFilter::AllowList(list) = &mut settings_with_message.field_printing {
                list.insert("message".into());
            }
            for ev in &self.events {
                let mut key = ev.to_string(&settings_with_message);
                key.truncate(truncated_key_width);
                if left_offset >= 1 {
                    write!(out, "{}", " ".repeat(left_offset - 1))?;
                }
                write!(out, ">{:width$}", key, width = truncated_key_width)?;
                let event_offset = (width(
                    render_conf.chart_width(),
                    render_conf.total(),
                    ev.timestamp.duration_since(span_info.start).unwrap(),
                ) as i32)
                    - 1;
                write!(out, "{}", " ".repeat(DURATION_WIDTH + 2))?;
                writeln!(
                    out,
                    "{}┼",
                    " ".repeat(base_offset + event_offset.max(0) as usize)
                )?;
            }
        }
        let map = &self.children;
        let mut children = map.values().collect::<Vec<_>>();
        children.sort_by_key(|child| child.info.as_ref().map(|it| it.start));

        for child in children {
            child._dump(
                out,
                render_conf,
                settings,
                left_offset + NESTED_EVENT_OFFSET,
            )?;
        }
        Ok(())
    }

    fn _with_tracker(
        &mut self,
        span: &span::Id,
        mut path: impl Iterator<Item = span::Id>,
        f: impl FnOnce(&mut SpanTracker),
    ) {
        match path.next() {
            None => f(self),
            Some(id) => {
                let child = self
                    .children
                    .entry(id.clone())
                    .or_insert_with(|| SpanTracker::new(id, None));
                child._with_tracker(span, path, f);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Types {
    events: bool,
    spans: bool,
}

/// Settings for [`TeXRayLayer`] output
#[derive(Clone, Debug)]
pub struct Settings {
    width: usize,
    min_duration: Option<Duration>,
    types: Types,
    field_printing: FieldFilter,
    updated: bool,
    out: DynWriter,
}

/// Wrap a dyn writer to get a Debug implementation
#[derive(Clone)]
struct DynWriter {
    inner: Arc<Mutex<dyn std::io::Write + Send>>,
}

impl Debug for DynWriter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn Writer")
    }
}

/// Filter to control which fields are printed along with spans
#[derive(Clone, Debug)]
enum FieldFilter {
    AllowList(HashSet<Cow<'static, str>>),
    DenyList(HashSet<Cow<'static, str>>),
}

impl Default for FieldFilter {
    fn default() -> Self {
        Self::DenyList(HashSet::new())
    }
}

impl FieldFilter {
    fn should_print(&self, field: &str) -> bool {
        match &self {
            FieldFilter::DenyList(deny) => !deny.contains(field),
            FieldFilter::AllowList(allow) => allow.contains(field),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            width: 120,
            min_duration: None,
            types: Types {
                events: false,
                spans: true,
            },
            field_printing: Default::default(),
            updated: false,
            out: DynWriter {
                inner: Arc::new(Mutex::new(stderr())),
            },
        }
    }
}

impl Settings {
    /// Load settings via term-size & defaults
    ///
    /// By default:
    /// - All fields are printed
    /// - Only spans are printed, events are not
    /// - Spans of any duration are printed
    pub fn auto() -> Self {
        let mut base = Settings::default();
        if let Some((w, _h)) = term_size::dimensions() {
            base.width = w;
        };
        base
    }

    /// Set the max-width when printing output
    #[must_use]
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self.updated = true;
        self
    }

    /// Overwrite the writer [`TexRayLayer`] will output to
    pub fn writer<W: Write + Send + 'static>(mut self, w: W) -> Self {
        self.out = DynWriter {
            inner: Arc::new(Mutex::new(w)),
        };
        self
    }

    /// Print events in addition to spans
    #[must_use]
    pub fn enable_events(mut self) -> Self {
        self.types.events = true;
        self.updated = true;
        self
    }

    /// When printing spans & events, only render the following fields
    #[must_use]
    pub fn only_show_fields(mut self, fields: &[&'static str]) -> Self {
        self.field_printing =
            FieldFilter::AllowList(fields.iter().map(|item| Cow::Borrowed(*item)).collect());
        self
    }

    /// When printing spans & events, print all fields, except for these fields
    #[must_use]
    pub fn suppress_fields(mut self, fields: &[&'static str]) -> Self {
        self.field_printing =
            FieldFilter::DenyList(fields.iter().map(|item| Cow::Borrowed(*item)).collect());
        self
    }

    /// Only show spans longer than a minimum duration
    #[must_use]
    pub fn min_duration(mut self, duration: Duration) -> Self {
        self.min_duration = Some(duration);
        self
    }
}

/// Tracing Layer to display a summary of spans.
///
/// _Note:_ This layer does nothing on its own. It must be used in combination with [`examine`] to
/// print the summary of a specific span.
#[derive(Clone, Debug)]
pub struct TeXRayLayer {
    tracked_spans: Arc<RwLock<HashMap<span::Id, SpanTracker>>>,
    settings: Settings,
}

/// Initialize a default subscriber and install it as the global default
pub fn init() {
    let layer = TeXRayLayer::new();
    use tracing_subscriber::layer::SubscriberExt;
    let registry = Registry::default().with(layer);
    tracing::subscriber::set_global_default(registry).expect("failed to install subscriber");
}

impl TeXRayLayer {
    fn _new() -> Self {
        Self {
            tracked_spans: Default::default(),
            settings: Default::default(),
        }
    }

    /// Create a new [`TeXRayLayer`] with settings from [`Settings::auto`]
    pub fn new() -> Self {
        let mut dumper = DUMPER.clone();
        if !dumper.settings.updated {
            dumper.settings = Settings::auto();
        }
        dumper
    }

    fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Show events in output in addition to spans
    pub fn enable_events(self) -> Self {
        Self {
            settings: self.settings.enable_events(),
            ..self
        }
    }

    /// Override the rendered width
    ///
    /// By default, the width is loaded by inspecting the TTY. If a TTY is not available,
    /// it defaults to 120
    pub fn width(mut self, width: usize) -> Self {
        self.settings.width = width;
        self
    }

    /// When printing spans & events, only render the following fields
    pub fn only_show_fields(mut self, fields: &[&'static str]) -> Self {
        self.settings = self.settings.only_show_fields(fields);
        self
    }

    /// Only render spans longer than `duration`
    pub fn min_duration(mut self, duration: Duration) -> Self {
        self.settings = self.settings.min_duration(duration);
        self
    }

    /// Create a [`TexRayLayer`] from specific settings
    pub fn with_settings(settings: Settings) -> Self {
        let mut layer = Self::_new();
        layer.configure(settings);
        layer
    }

    /// Update the settings of this [`TexRayLayer`]
    pub fn update_settings(mut self, f: impl Fn(Settings) -> Settings) -> Self {
        self.settings = f(self.settings);
        self
    }

    fn configure(&mut self, settings: Settings) {
        self.settings = settings;
        self.settings.updated = true;
    }

    fn for_relevant_trackers<'a, S>(
        &self,
        span: &span::Id,
        ctx: &Context<'a, S>,
        f: impl Fn(&mut SpanTracker),
    ) where
        S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
    {
        if let Some(span_iter) = ctx.span_scope(span) {
            let mut tracker = self.tracked_spans.write().unwrap();
            for span_ref in span_iter {
                if let Some(inner) = tracker.get_mut(&span_ref.id()) {
                    f(inner)
                }
            }
        }
    }

    fn dump_on_exit(&self, span: &Span, settings: Option<Settings>) {
        if let Some(id) = span.id() {
            self.tracked_spans
                .write()
                .unwrap()
                .insert(id.clone(), SpanTracker::new(id, settings));
        }
    }
}

fn pretty_duration(duration: Duration) -> String {
    const NANOS_PER_SEC: u128 = 1_000_000_000;
    let divisors = [
        ("m ", (60 * NANOS_PER_SEC)),
        ("s ", NANOS_PER_SEC),
        ("ms", NANOS_PER_SEC / 1000),
        ("μs", NANOS_PER_SEC / 1000 / 1000),
        ("ns", 1),
    ];
    let nanos = duration.as_nanos();
    if nanos == 0 {
        return "0ns".to_string();
    }
    for (unit, div) in divisors {
        if nanos / div > 1 {
            return format!("{}{}", nanos / div, unit);
        }
    }
    unreachable!("{:?}", duration)
}

impl<S> Layer<S> for TeXRayLayer
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    fn on_event(&self, event: &TracingEvent<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.current_span().id() {
            let mut metadata = TrackedMetadata::default();
            event.record(&mut metadata);
            let tracked_event = EventInfo {
                timestamp: SystemTime::now(),
                metadata,
            };
            self.for_relevant_trackers(span, &ctx, |tracker| {
                tracker.add_event(span, tracker.path(span, &ctx), tracked_event.clone())
            })
        }
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        self.for_relevant_trackers(id, &ctx, |tracker| {
            tracker.record_metadata(id, tracker.path(id, &ctx), attrs)
        })
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        self.for_relevant_trackers(id, &ctx, |tracker| {
            tracker.record_metadata(id, tracker.path(id, &ctx), values);
        })
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        self.for_relevant_trackers(id, &ctx, |tracker| {
            tracker.open(id, tracker.path(id, &ctx), SpanInfo::for_span(id, &ctx));
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        self.for_relevant_trackers(&id, &ctx, |tracker| {
            tracker.exit(&id, tracker.path(&id, &ctx), SystemTime::now())
        });
        if let Some(tracker) = self.tracked_spans.read().unwrap().get(&id) {
            let _ = tracker.dump(self.settings());
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{width, Settings, SpanInfo, SpanTracker};
    use std::iter;
    use std::ops::Add;
    use std::time::{Duration, UNIX_EPOCH};
    use tracing::Id;

    #[test]
    fn compute_relative_width() {
        let total = Duration::from_secs(10);
        let partial = Duration::from_secs(1);
        assert_eq!(width(10, total, partial), 1);

        let total = Duration::from_secs(10);
        let partial = Duration::from_secs_f64(2.9);
        assert_eq!(width(10, total, partial), 3);

        let total = Duration::from_secs_f64(0.045532);
        let partial = Duration::from_secs_f64(0.034389);
        assert_eq!(width(120, total, partial), 91);
        let total = Duration::from_secs_f64(0.045532);
        let partial = Duration::from_secs_f64(0.034489);
        assert_eq!(width(120, total, partial), 91);
    }

    fn validate_output(max_width: usize, output: &str) {
        for line in output.lines() {
            assert!(
                line.chars().count() <= max_width,
                "`{}` was too long ({} > {})",
                line,
                line.chars().count(),
                max_width
            )
        }
    }

    fn dump_to_string(tracker: &SpanTracker) -> String {
        let mut out = vec![];
        let settings = Settings::default();
        tracker.dump_to(&mut out, &settings).unwrap();
        String::from_utf8(out.clone()).unwrap()
    }

    #[test]
    fn render_correct_output() {
        let id_0 = Id::from_u64(1);
        let id_1 = Id::from_u64(2);
        let mut tracker = SpanTracker::new(id_0.clone(), None);
        let interval_start = UNIX_EPOCH;
        let interval_end = UNIX_EPOCH.add(Duration::from_secs(10));
        tracker.open(
            &id_0,
            iter::empty(),
            SpanInfo {
                name: "test".to_string(),
                start: interval_start,
                end: None,
            },
        );
        {
            tracker.open(
                &id_1,
                &mut [id_1.clone()].iter().cloned(),
                SpanInfo {
                    name: "nested".to_string(),
                    start: interval_start + Duration::from_secs(2),
                    end: None,
                },
            );
            tracker.exit(
                &id_1,
                &mut [id_1.clone()].iter().cloned(),
                interval_start + Duration::from_secs(7),
            );
        }
        tracker.exit(&id_0, iter::empty(), interval_end);
        let settings = Settings::default();
        let output = dump_to_string(&tracker);
        assert_eq!(output, r#"
test       10s  ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  nested    5s                       ├──────────────────────────────────────────────────┤
"#.trim_start());
        validate_output(settings.width, &output);
    }
}
