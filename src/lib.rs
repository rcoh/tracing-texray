#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    rustdoc::missing_crate_level_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

mod tracked_spans;
mod tracker;

use parking_lot::Mutex;
use std::borrow::Cow;

use parking_lot::lock_api::{MutexGuard, RawMutex};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::io::{stderr, Write};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tracing::span::{Attributes, Record};

use tracing::subscriber::set_global_default;
use tracing::{Event as TracingEvent, Id, Span, Subscriber};

use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};
use tracker::{EventInfo, FieldSettings, InterestTracker, RootTracker, SpanInfo, TrackedMetadata};

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
    field_filter: FieldFilter,
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
            fields: FieldSettings::new(self.field_filter),
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
            field_filter: Default::default(),
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
        self.field_filter =
            FieldFilter::AllowList(fields.iter().map(|item| Cow::Borrowed(*item)).collect());
        self
    }

    /// When printing spans & events, print all fields, except for these fields
    #[must_use]
    pub fn suppress_fields(mut self, fields: &[&'static str]) -> Self {
        self.field_filter =
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

                event.record(&mut tracker.field_recorder(&mut metadata));
                let tracked_event = EventInfo::now(metadata);
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
            if self.tracker.end_tracking(id.clone()) {
                let _ = tracker
                    .dump()
                    .map_err(|err| eprintln!("failed to dump output: {}", err));
            }
        });
    }
}
