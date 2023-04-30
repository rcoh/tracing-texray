use crate::test_util::CaptureWriter;
use std::time::Duration;
use tracing::{info_span, trace_span};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::EnvFilter;
use tracing_texray::TeXRayLayer;

//#[tokio::test]
#[test]
fn test_me() {
    env_logger::init();
    let capture_writer = CaptureWriter::stdout();
    let layer = TeXRayLayer::new()
        .width(80)
        .enable_events()
        .update_settings(|s| s.writer(capture_writer.clone()));
    let registry = tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(layer);
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

    for _ in 0..2 {
        somewhere_deep_in_my_program();
        /*tokio::spawn(async {
            somewhere_deep_in_my_program();
        })
        .await
        .unwrap();*/
    }

    assert!(
        capture_writer.to_string().contains(">buzz"),
        "event not received! {}",
        capture_writer
    );
}

fn somewhere_deep_in_my_program() {
    for id in 10..500000 {
        some_other_function(id);
    }
    //some_other_function(1000000);
    tracing_texray::examine(info_span!("do_a_thing")).in_scope(|| {
        for id in 0..5 {
            some_other_function(id);
        }
    });
}

fn some_other_function(id: usize) {
    trace_span!("inner_task", id = %id).in_scope(|| tracing::info!("buzz"));
    // ...
}

mod test_util {
    use parking_lot::Mutex;
    use std::fmt::{Display, Formatter};
    use std::io::{stdout, Write};
    use std::sync::Arc;

    #[derive(Clone)]
    pub struct CaptureWriter {
        inner: Arc<Mutex<dyn Write + Send>>,
        data: Arc<Mutex<Vec<u8>>>,
    }

    impl Display for CaptureWriter {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                std::str::from_utf8(self.data.lock().as_ref()).unwrap()
            )
        }
    }

    impl CaptureWriter {
        pub fn stdout() -> Self {
            CaptureWriter {
                inner: Arc::new(Mutex::new(stdout())),
                data: Default::default(),
            }
        }
    }

    impl Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.data.lock().extend(buf);
            self.inner.lock().write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.lock().flush()
        }
    }
}
