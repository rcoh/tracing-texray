# tracing-texray

`tracing-texray` is a tracing layer to introspect tracing spans and events in plain text. By `examine`-ing
a specific span, a full tree will be dumped when that span exits. Example output:

```text
load_data                                52ms ├────────────────────────────────┤
  download_results{uri: www.crates.io}   11ms                ├─────┤
  >URI resolved                                              ┼
  >connected                                                    ┼
  compute_stats                          10ms                        ├─────┤
  render_response                         6ms                               ├──┤
```

In cases where a more powerful solution like [tracing-chrome](https://crates.io/crates/tracing-chrome) is not required,
`tracing-texray` gives enables getting a lightweight timeline of what happened when.

## Usage
`tracing-xray` combines two pieces: a global subscriber, and local span examination. 

**First**, the layer must be installed globally:
```rust,no_run
use std::time::Duration;
use tracing_texray::TeXRayLayer;
use tracing_subscriber::{Registry, EnvFilter, layer::SubscriberExt};
fn main() {
    // Option A: Exclusively using tracing_texray:
    tracing_texray::init();
    
    // Option B: install the layer in combination with other layers, eg. tracing_subscriber::fmt:
    let subscriber = Registry::default()
        .with(EnvFilter::try_from_default_env().expect("invalid env filter"))
        .with(tracing_subscriber::fmt::layer())
        .with(
            TeXRayLayer::new()
                // by default, all metadata fields will be printed. If this is too noisy,
                // fitler only the fields you care about
                .only_show_fields(&["name", "operation", "service"])
                // only print spans longer than a certain duration
                .min_duration(Duration::from_millis(100)),
        );
    tracing::subscriber::set_global_default(subscriber).unwrap();
}
```

**Next**, wrap any spans you want to track with `examine`:
```rust
use tracing::info_span;
use tracing_texray::examine;
fn somewhere_deep_in_my_program() {
    examine(info_span!("do_a_thing")).in_scope(|| {
        some_other_function();
    })
}

fn some_other_function() {
    // ...
}
```