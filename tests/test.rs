use std::time::Duration;
use tracing::info_span;

#[test]
fn test_me() {
    tracing_texray::init();
    tracing_texray::examine_with(
        info_span!("load_data"),
        tracing_texray::Settings::default()
            .enable_events()
            .width(80),
    )
    .in_scope(|| {
        std::thread::sleep(Duration::from_millis(20));
        info_span!("download_results", uri = %"www.crates.io").in_scope(|| {
            tracing::info!("URI resolved");
            std::thread::sleep(Duration::from_millis(5));
            tracing::info!("connected");
            std::thread::sleep(Duration::from_millis(5));
        });
        info_span!("compute_stats").in_scope(|| {
            std::thread::sleep(Duration::from_millis(10));
        });
        info_span!("render_response").in_scope(|| {
            std::thread::sleep(Duration::from_millis(5));
        })
    });
}
