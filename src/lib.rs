#![doc = include_str ! ("../README.md")]
#![warn(
    missing_docs,
    rustdoc::missing_crate_level_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

mod hashset;

use parking_lot::{Mutex, RwLock};
use std::borrow::Cow;

use parking_lot::lock_api::{MutexGuard, RawMutex};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::{stderr, Write};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::{io, iter};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};

use tracing::subscriber::set_global_default;
use tracing::{Event as TracingEvent, Id, Span, Subscriber};
use tracing_subscriber::field::RecordFields;

use crate::hashset::TrackedSpans;
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
    fn to_string(&self, settings: &FieldSettings) -> String {
        let mut out = String::new();
        self.metadata
            .write(&mut out, settings)
            .expect("writing to a string cannot fail");
        out
    }
}

impl SpanInfo {
    fn for_span<S>(span: &Id, ctx: &Context<'_, S>) -> Self
    where
        S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
    {
        Self {
            name: ctx
                .metadata(span)
                .map(|metadata| metadata.name())
                .unwrap_or("could-not-find-span"),
            start: SystemTime::now(),
            end: None,
        }
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_TEXRAY_LAYER: TeXRayLayer = TeXRayLayer::uninitialized();
}

macro_rules! check_initialized {
    ($self: expr) => {
        if !$self.initialized() {
            return;
        }
    };
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
    GLOBAL_TEXRAY_LAYER.dump_on_exit(&span, Some(local_settings.locked()));
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
    GLOBAL_TEXRAY_LAYER.dump_on_exit(&span, None);
    span
}

#[derive(Default, Debug, Clone)]
struct TrackedMetadata {
    data: Vec<(&'static str, String)>,
}

impl TrackedMetadata {
    fn write(&self, f: &mut impl std::fmt::Write, settings: &FieldSettings) -> std::fmt::Result {
        let relevant_fields = || {
            self.data
                .iter()
                .filter(|(f, _)| settings.field_printing.should_print(f))
        };

        if let Some((_, message)) = relevant_fields().find(|(k, _)| *k == "message") {
            write!(f, "{}", message.lines().next().unwrap_or_default())?;
        }

        let relevant_fields = || relevant_fields().filter(|(k, _v)| *k != "message");

        if relevant_fields().count() == 0 {
            return Ok(());
        }

        write!(f, "{{")?;
        let mut peekable = relevant_fields().peekable();
        while let Some((k, v)) = peekable.next() {
            write!(f, "{}: {}", k, v)?;
            if peekable.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

#[derive(Debug, Clone)]
struct SpanInfo {
    start: SystemTime,
    end: Option<SystemTime>,
    name: &'static str,
}

impl SpanInfo {
    fn full_name(&self, tracker: &SpanTracker, settings: &FieldSettings) -> String {
        let mut id = self.name.to_string();
        tracker
            .metadata
            .write(&mut id, settings)
            .expect("rendering to string cannot fail");
        id
    }

    fn duration(&self) -> Option<Duration> {
        self.end.and_then(|end| end.duration_since(self.start).ok())
    }

    fn render(
        &self,
        out: &mut dyn Write,
        tracker: &SpanTracker,
        settings: &RenderSettings,
        render_conf: &RenderConf,
        left_offset: usize,
    ) -> io::Result<()> {
        let mut key = self.full_name(tracker, &tracker.settings);
        let truncated_key_width = render_conf.key_width - left_offset;
        key.truncate(truncated_key_width);
        let ev_start_ts = self.start;
        let span_len = match self.duration() {
            None => return Ok(()),
            Some(dur) => dur,
        };
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
            match ev_start_ts.duration_since(render_conf.start_ts) {
                Ok(dur) => dur,
                Err(_) => return Ok(()),
            },
        );
        write!(out, "{}", " ".repeat(offset))?;
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

/// Tracker of an individual span
#[derive(Debug)]
struct SpanTracker {
    info: Option<SpanInfo>,
    metadata: TrackedMetadata,
    events: Vec<EventInfo>,
    settings: Arc<FieldSettings>,
}

struct FieldFilterTracked<'a> {
    field_filter: &'a FieldFilter,
    tracked_metadata: &'a mut TrackedMetadata,
}

impl Visit for FieldFilterTracked<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if self.field_filter.should_print(field.name()) {
            self.tracked_metadata
                .data
                .push((field.name(), format!("{:?}", value)));
        }
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
        // start_ts is always less than end_ts
        self.end_ts
            .duration_since(self.start_ts)
            .unwrap_or_default()
    }

    fn chart_width(&self) -> usize {
        self.width
            .checked_sub(self.key_width)
            .and_then(|w| w.checked_sub(DURATION_WIDTH + 2))
            .unwrap_or(20)
    }
}

impl SpanTracker {
    fn new(settings: Arc<FieldSettings>) -> Self {
        Self {
            info: None,
            events: vec![],
            metadata: Default::default(),
            settings,
        }
    }

    fn record_metadata(&mut self, values: &dyn RecordFields) {
        values.record(&mut FieldFilterTracked {
            field_filter: &self.settings.field_printing,
            tracked_metadata: &mut self.metadata,
        });
    }

    fn add_event(&mut self, event: EventInfo) {
        self.events.push(event)
    }

    fn open(&mut self, span_info: SpanInfo) {
        match self.info {
            None => self.info = Some(span_info),
            Some(_) => {} // already open, don't update
        }
    }

    fn exit(&mut self, timestamp: SystemTime) {
        match &mut self.info {
            Some(info) => info.end = Some(timestamp),
            None => eprintln!("this is a bug"), //
        }
    }

    fn span_info(&self) -> impl Iterator<Item = &SpanInfo> {
        self.info.iter()
    }

    fn max_key_width(&self, depth: usize) -> usize {
        let longest_self = self
            .info
            .as_ref()
            .map(|info| info.full_name(self, &self.settings).len())
            .unwrap_or_default();
        longest_self + NESTED_EVENT_OFFSET * (depth - 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    default_output: DynWriter,
}

impl Settings {
    fn locked(self) -> SpanSettings {
        SpanSettings {
            render: RenderSettings {
                width: self.width,
                min_duration: self.min_duration,
                types: self.types,
            },
            fields: FieldSettings {
                field_printing: self.field_printing,
            },
            out: self.default_output,
        }
    }
}

/// Wrap a dyn writer to get a Debug implementation
#[derive(Clone)]
struct DynWriter {
    inner: Arc<Mutex<dyn Write + Send>>,
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
        if field == "message" {
            return true;
        }
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
            default_output: DynWriter {
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
        self
    }

    /// Overwrite the writer [`TexRayLayer`] will output to
    pub fn writer<W: Write + Send + 'static>(&mut self, w: W) -> &mut Self {
        self.default_output = DynWriter {
            inner: Arc::new(Mutex::new(w)),
        };
        self
    }

    /// Print events in addition to spans
    pub fn enable_events(mut self) -> Self {
        self.set_enable_events(true);
        self
    }

    fn set_enable_events(&mut self, enabled: bool) -> &mut Self {
        self.types.events = enabled;
        self
    }

    /// When printing spans & events, only render the following fields
    pub fn only_show_fields(&mut self, fields: &[&'static str]) -> &mut Self {
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
    pub fn min_duration(&mut self, duration: Duration) -> &mut Self {
        self.min_duration = Some(duration);
        self
    }
}

/// Tracing Layer to display a summary of spans.
///
/// _Note:_ This layer does nothing on its own. It must be used in combination with [`examine`] to
/// print the summary of a specific span.
#[derive(Debug, Clone)]
pub struct TeXRayLayer {
    settings: Arc<Mutex<SettingsContainer>>,
    initialized: Arc<AtomicBool>,
    tracker: Arc<RootTracker>,
}

impl Default for TeXRayLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct RootTracker {
    root_span_ids: TrackedSpans,
    span_metadata: RwLock<HashMap<Id, InterestTracker>>,
}

#[derive(Debug)]
struct InterestTracker {
    field_settings: Arc<FieldSettings>,
    render_settings: RenderSettings,
    out: DynWriter,
    children: HashMap<Vec<Id>, SpanTracker>,
}

impl InterestTracker {
    fn new(
        id: Id,
        settings: RenderSettings,
        field_settings: FieldSettings,
        out: DynWriter,
    ) -> Self {
        let mut children = HashMap::new();
        children.insert(vec![id], SpanTracker::new(Arc::new(field_settings.clone())));
        Self {
            children,
            field_settings: Arc::new(field_settings),
            render_settings: settings,
            out,
        }
    }

    fn new_span(&mut self, path: Vec<Id>) -> &mut SpanTracker {
        debug_assert!(!path.is_empty());
        let settings = self.field_settings.clone();
        self.children
            .entry(path)
            .or_insert_with(|| SpanTracker::new(settings))
    }

    fn record_metadata(&mut self, path: &[Id], fields: &dyn RecordFields) {
        if let Some(s) = self.children.get_mut(path) {
            s.record_metadata(fields)
        }
    }

    #[track_caller]
    fn span(&mut self, path: Vec<Id>) -> &mut SpanTracker {
        debug_assert!(!path.is_empty());
        let settings = self.field_settings.clone();
        self.children
            .entry(path)
            .or_insert_with(|| SpanTracker::new(settings))
    }

    fn open(&mut self, path: Vec<Id>, span_info: SpanInfo) {
        self.span(path).open(span_info);
    }

    fn add_event(&mut self, path: Vec<Id>, event: EventInfo) {
        self.span(path).add_event(event);
    }

    fn exit(&mut self, path: Vec<Id>, timestamp: SystemTime) {
        self.span(path).exit(timestamp);
    }

    fn spans(&self) -> impl Iterator<Item = &SpanInfo> {
        self.children.values().flat_map(|c| c.span_info())
    }

    fn dump(&self) -> io::Result<()> {
        let mut out = self.out.inner.lock();
        let settings = &self.render_settings;
        let all_events = self.spans().collect::<Vec<_>>();
        if all_events.is_empty() {
            write!(&mut out, "no events...")?;
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
            key_width: self
                .children
                .iter()
                .map(|(path, t)| t.max_key_width(path.len()))
                .max()
                .unwrap_or(120)
                .min(120),
            width: settings.width,
        };
        let mut ordered = self.children.iter().collect::<Vec<_>>();
        ordered
            .sort_by_key(|(key, _)| sort_key(&self.children, key.as_slice()).collect::<Vec<_>>());

        for (key, track) in ordered.iter() {
            let offset = NESTED_EVENT_OFFSET * (key.len() - 1);
            if let Some(info) = track.info.as_ref() {
                if settings.types.spans {
                    info.render(out.deref_mut(), track, settings, &conf, offset)?;
                }
                if settings.types.events {
                    self.render_events(
                        out.deref_mut(),
                        &track.events,
                        offset,
                        &conf,
                        info,
                        &self.field_settings,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn render_events(
        &self,
        mut out: impl Write,
        events: &[EventInfo],
        left_offset: usize,
        render_conf: &RenderConf,
        span_info: &SpanInfo,
        field_settings: &FieldSettings,
    ) -> io::Result<()> {
        let left_offset = left_offset + 2;
        let truncated_key_width = render_conf.key_width - left_offset;
        let base_offset = width(
            render_conf.chart_width(),
            render_conf.total(),
            span_info
                .start
                .duration_since(render_conf.start_ts)
                .expect("start_ts MUST be before span_info.start because it is a minima"),
        );
        let mut settings_with_message = field_settings.clone();
        if let FieldFilter::AllowList(list) = &mut settings_with_message.field_printing {
            list.insert("message".into());
        }
        for ev in events {
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
        Ok(())
    }
}

impl RootTracker {
    fn new() -> Self {
        Self {
            root_span_ids: TrackedSpans::new(1024),
            span_metadata: Default::default(),
        }
    }

    fn register_interest(&self, id: Id, settings: SpanSettings) {
        self.root_span_ids.insert(id.into_non_zero_u64());
        self.span_metadata.write().insert(
            id.clone(),
            InterestTracker::new(id, settings.render, settings.fields, settings.out),
        );
    }

    fn if_interested<T>(
        &self,
        ids: impl Iterator<Item = Id>,
        f: impl Fn(&mut InterestTracker, &mut dyn Iterator<Item = Id>) -> T,
    ) -> Option<T> {
        let mut iter = ids.skip_while(|id| !self.root_span_ids.contains(id.into_non_zero_u64()));
        if let Some(root) = iter.next() {
            assert!(self.root_span_ids.contains(root.into_non_zero_u64()));
            let mut tracker = self.span_metadata.write();
            let mut with_root = iter::once(root.clone()).chain(iter);
            Some(f(
                tracker.get_mut(&root).expect("must exist"),
                &mut with_root,
            ))
        } else {
            None
        }
    }
}

fn sort_key(
    map: &HashMap<Vec<Id>, SpanTracker>,
    target: &[Id],
) -> impl Iterator<Item = SystemTime> {
    if target.is_empty() {
        Box::new(std::iter::empty()) as Box<dyn Iterator<Item = SystemTime>>
    } else {
        let span = map.get(target).expect("missing");
        Box::new(
            std::iter::once(span.info.as_ref().unwrap().start)
                .chain(sort_key(map, &target[..target.len() - 1])),
        )
    }
}

#[derive(Debug)]
enum SettingsContainer {
    Unlocked(Settings),
    Locked(SpanSettings),
}

#[derive(Debug, Clone)]
struct SpanSettings {
    render: RenderSettings,
    fields: FieldSettings,
    out: DynWriter,
}

#[derive(Debug, Clone)]
struct RenderSettings {
    width: usize,
    min_duration: Option<Duration>,
    types: Types,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 120,
            min_duration: None,
            types: Types {
                events: false,
                spans: true,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct FieldSettings {
    field_printing: FieldFilter,
}

impl Default for FieldSettings {
    fn default() -> Self {
        Self {
            field_printing: FieldFilter::DenyList(HashSet::new()),
        }
    }
}

impl SettingsContainer {
    fn lock_settings(&mut self) -> &SpanSettings {
        match self {
            SettingsContainer::Locked(settings) => settings,
            SettingsContainer::Unlocked(settings) => {
                let cloned = settings.clone();
                *self = SettingsContainer::Locked(cloned.locked());
                self.lock_settings()
            }
        }
    }
    fn settings_mut(&mut self) -> Option<&mut Settings> {
        match self {
            SettingsContainer::Unlocked(settings) => Some(settings),
            SettingsContainer::Locked(_) => None,
        }
    }
}

impl Default for SettingsContainer {
    fn default() -> Self {
        SettingsContainer::Unlocked(Settings::default())
    }
}

/// Initialize a default subscriber and install it as the global default
pub fn init() {
    let layer = TeXRayLayer::new();
    use tracing_subscriber::layer::SubscriberExt;
    let registry = Registry::default().with(layer);
    set_global_default(registry).expect("failed to install subscriber");
}

impl TeXRayLayer {
    fn uninitialized() -> Self {
        Self {
            settings: Default::default(),
            initialized: Arc::new(AtomicBool::new(false)),
            tracker: Arc::new(RootTracker::new()),
        }
    }

    /// Create a new [`TeXRayLayer`] with settings from [`Settings::auto`]
    pub fn new() -> Self {
        let dumper = GLOBAL_TEXRAY_LAYER.clone();
        dumper.initialized.store(true, Ordering::Relaxed);
        *dumper.settings.lock() = SettingsContainer::Unlocked(Settings::auto());
        dumper
    }

    pub(crate) fn settings_mut(&mut self) -> impl DerefMut<Target = Settings> + '_ {
        struct DerefSettings<'a, R: RawMutex> {
            target: MutexGuard<'a, R, SettingsContainer>,
        }
        impl<'a, R: RawMutex> Deref for DerefSettings<'a, R> {
            type Target = Settings;

            fn deref(&self) -> &Self::Target {
                match self.target.deref() {
                    SettingsContainer::Unlocked(s) => s,
                    SettingsContainer::Locked(_s) => panic!(),
                }
            }
        }
        impl<'a, R: RawMutex> DerefMut for DerefSettings<'a, R> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                self.target
                    .deref_mut()
                    .settings_mut()
                    .expect("cannot modify settings when already in progress")
            }
        }
        DerefSettings {
            target: self.settings.lock(),
        }
    }

    /// Install [`TexRayLayer`] as the global default
    pub fn init(self) {
        let registry = tracing_subscriber::registry().with(self);
        use tracing_subscriber::layer::SubscriberExt;
        set_global_default(registry).expect("failed to install subscriber")
    }

    /// Show events in output in addition to spans
    pub fn enable_events(mut self) -> Self {
        self.settings_mut().set_enable_events(true);
        self
    }

    /// Override the rendered width
    ///
    /// By default, the width is loaded by inspecting the TTY. If a TTY is not available,
    /// it defaults to 120
    pub fn width(mut self, width: usize) -> Self {
        self.settings_mut().width = width;
        self
    }

    /// When printing spans & events, only render the following fields
    pub fn only_show_fields(mut self, fields: &[&'static str]) -> Self {
        self.settings_mut().only_show_fields(fields);
        self
    }

    /// Only render spans longer than `duration`
    pub fn min_duration(mut self, duration: Duration) -> Self {
        self.settings_mut().min_duration(duration);
        self
    }

    /// Update the settings of this [`TexRayLayer`]
    pub fn update_settings(mut self, f: impl FnOnce(&mut Settings) -> &mut Settings) -> Self {
        // TODO: assert!(self.tracked_spans_v2.is_empty());
        f(self.settings_mut().deref_mut());
        self
    }

    /// Updates the `writer` used to dump output
    pub fn writer(self, writer: impl Write + Send + 'static) -> Self {
        self.update_settings(move |s| s.writer(writer))
    }

    fn for_tracker<'a, S>(
        &self,
        span: &Id,
        ctx: &Context<'a, S>,
        f: impl Fn(&mut InterestTracker, Vec<Id>),
    ) where
        S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
    {
        if let Some(path) = ctx.span_scope(span) {
            self.tracker
                .if_interested(path.from_root().map(|s| s.id()), |tracker, path| {
                    f(tracker, path.collect::<Vec<_>>())
                });
        }
    }

    fn dump_on_exit(&self, span: &Span, settings: Option<SpanSettings>) {
        check_initialized!(self);
        if let Some(id) = span.id() {
            self.tracker.register_interest(
                id,
                settings.unwrap_or(self.settings.lock().lock_settings().clone()),
            );
        }
    }

    fn initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
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
        if nanos / div >= 1 {
            return format!("{}{}", nanos / div, unit);
        }
    }
    unreachable!("{:?}", duration)
}

impl<S> Layer<S> for TeXRayLayer
where
    S: Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        check_initialized!(self);
        self.for_tracker(id, &ctx, |tracker, path| {
            tracker.new_span(path).record_metadata(attrs);
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        check_initialized!(self);
        self.for_tracker(id, &ctx, |tracker, path| {
            tracker.record_metadata(&path, values)
        })
    }

    fn on_event(&self, event: &TracingEvent<'_>, ctx: Context<'_, S>) {
        check_initialized!(self);
        if let Some(span) = ctx.current_span().id() {
            self.for_tracker(span, &ctx, |tracker, path| {
                let mut metadata = TrackedMetadata::default();

                event.record(&mut FieldFilterTracked {
                    field_filter: &tracker.field_settings.field_printing,
                    tracked_metadata: &mut metadata,
                });
                let tracked_event = EventInfo {
                    timestamp: SystemTime::now(),
                    metadata,
                };
                tracker.add_event(path, tracked_event);
            });
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        check_initialized!(self);
        self.for_tracker(id, &ctx, |tracker, path| {
            tracker.open(path, SpanInfo::for_span(id, &ctx));
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        check_initialized!(self);
        self.for_tracker(&id, &ctx, |tracker, path| {
            tracker.exit(path, SystemTime::now());
            if self.tracker.root_span_ids.remove(id.into_non_zero_u64()) {
                let _ = tracker
                    .dump()
                    .map_err(|err| eprintln!("failed to dump output: {}", err));
            }
        });
    }
}

#[cfg(test)]
mod test {
    use crate::{
        width, DynWriter, FieldSettings, InterestTracker, RenderSettings, Settings, SpanInfo,
        TrackedMetadata,
    };
    use std::io::{BufWriter, Write};

    use std::mem::take;

    use std::ops::Add;
    use std::sync::Arc;

    use parking_lot::Mutex;
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

    fn dump_to_string(id: Id, f: impl Fn(&mut InterestTracker)) -> String {
        let settings = Settings::default();
        let settings = RenderSettings {
            width: settings.width,
            min_duration: settings.min_duration,
            types: settings.types,
        };
        let (writer, buf) = DynWriter::str();
        let mut tracker = InterestTracker::new(id, settings, FieldSettings::default(), writer);
        f(&mut tracker);
        tracker.dump().unwrap();
        let mut buf = buf.lock();
        buf.flush().unwrap();
        String::from_utf8(take(buf.get_mut())).unwrap()
    }

    #[test]
    fn render_metadata() {
        let metadata = TrackedMetadata {
            data: vec![("A", "B".to_string()), ("c", "d".to_string())],
        };
        let mut out = String::new();
        metadata.write(&mut out, &FieldSettings::default()).unwrap();
        assert_eq!(out, "{A: B, c: d}");
    }

    #[test]
    fn render_empty_metadata() {
        let metadata = TrackedMetadata { data: vec![] };
        let mut out = String::new();
        metadata.write(&mut out, &FieldSettings::default()).unwrap();
        assert_eq!(out, "");
    }

    impl DynWriter {
        fn str() -> (DynWriter, Arc<Mutex<BufWriter<Vec<u8>>>>) {
            let buf = Arc::new(Mutex::new(BufWriter::new(vec![])));
            (DynWriter { inner: buf.clone() }, buf)
        }
    }

    #[test]
    fn render_correct_output() {
        fn id(i: u64) -> Id {
            Id::from_u64(i)
        }
        let output = dump_to_string(id(1), |tracker| {
            let interval_start = UNIX_EPOCH;
            let interval_end = UNIX_EPOCH.add(Duration::from_secs(10));
            tracker.new_span(vec![id(1)]);
            tracker.new_span(vec![id(1), id(2)]);
            tracker.open(
                vec![id(1)],
                SpanInfo {
                    name: "test",
                    start: interval_start,
                    end: None,
                },
            );
            tracker.open(
                vec![id(1), id(2)],
                SpanInfo {
                    name: "nested",
                    start: interval_start + Duration::from_secs(2),
                    end: None,
                },
            );
            tracker.exit(vec![id(1), id(2)], interval_start + Duration::from_secs(7));
            tracker.exit(vec![id(1)], interval_end);
        });
        assert_eq!(output, r#"
test       10s  ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  nested    5s                       ├──────────────────────────────────────────────────┤
"#.trim_start());
    }
}
