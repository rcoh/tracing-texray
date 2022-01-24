use std::time::Duration;
use tracing::info_span;
use tracing_subscriber::layer::SubscriberExt;
use tracing_texray::TeXRayLayer;

#[tokio::test]
async fn test_me() {
    let layer = TeXRayLayer::new().width(80).enable_events();
    let registry = tracing_subscriber::registry().with(layer);
    tracing::subscriber::set_global_default(registry).expect("failed to install subscriber");
    tracing_texray::examine(info_span!("load_data")).in_scope(|| {
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

    for _ in 0..5 {
        tokio::spawn(async {
            somewhere_deep_in_my_program();
        }).await.unwrap();
    }
}

fn somewhere_deep_in_my_program() {
    tracing_texray::examine(info_span!("do_a_thing")).in_scope(|| {
        for id in 0..5 {
            some_other_function(id);
        }
    })
}

fn some_other_function(id: usize) {
    info_span!("inner_task", id = %id).in_scope(|| tracing::info!("buzz"));
    // ...
}
