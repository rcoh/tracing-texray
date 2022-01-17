use std::time::Duration;
use tracing::info_span;

#[test]
fn test_me() {
    tracing_texray::init();
    tracing_texray::examine_with(
        info_span!("load_data"),
        tracing_texray::Settings::auto().enable_events(),
    )
    .in_scope(|| {
        std::thread::sleep(Duration::from_millis(20));
        info_span!("download_results", uri = %"www.crates.io").in_scope(|| {
            tracing::info!("URI resolved");
            std::thread::sleep(Duration::from_millis(5));
            tracing::info!("data loaded");
            std::thread::sleep(Duration::from_millis(5));
        });
        info_span!("compute_stats").in_scope(|| {
            std::thread::sleep(Duration::from_millis(10));
        });
    });
}
