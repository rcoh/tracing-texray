use std::fmt::{Debug, Formatter};
use std::num::NonZeroU64;
use sync::{AtomicU64, Ordering};

mod sync {
    #[cfg(loom)]
    pub(crate) use loom::sync::atomic::{AtomicU64, Ordering};

    #[cfg(not(loom))]
    pub(crate) use std::sync::atomic::{AtomicU64, Ordering};
}

/// Lock-free hashset that can hold a fixed number of U64s
///
/// This allows tracking which spans are being examined without acquiring a lock
pub(crate) struct TrackedSpans {
    els: Vec<AtomicU64>,
}

impl Debug for TrackedSpans {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "debug")
    }
}

const EMPTY: u64 = 0;
const TOMBSTONE: u64 = u64::MAX;

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum InsertResult {
    AlreadyPresent,
    NotPresent,
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) struct MapFull;

impl TrackedSpans {
    pub(crate) fn new(max_spans: usize) -> Self {
        let mut storage = Vec::with_capacity(max_spans);
        for _ in 0..max_spans {
            storage.push(AtomicU64::new(EMPTY))
        }
        assert_eq!(storage.capacity(), max_spans);
        assert_eq!(storage.len(), max_spans);

        Self { els: storage }
    }

    fn hash(&self, value: u64, attempt: usize) -> usize {
        // to store the TOMBSTONE value, we reserve the final slot in the array to hold it,
        // if present
        if value == TOMBSTONE {
            if attempt != 0 {
                unreachable!("tombstone will never fail if missing")
            }
            self.els.len() - 1
        } else {
            ((value + attempt as u64) % (self.els.len() as u64 - 1)) as usize
        }
    }

    fn size(&self) -> usize {
        self.els.len() - 1
    }

    /// Insert a value into the array
    ///
    /// If the value was able to be inserted:
    /// - Some(false) will be returned if the value was already present
    /// - Some(true) will be returne
    pub(crate) fn insert(&self, value: NonZeroU64) -> Result<InsertResult, MapFull> {
        let value = value.get();
        let mut attempt = 0_usize;
        while attempt < self.size() {
            let idx = self.hash(value, attempt);
            let atomic = self.els.get(idx).expect("idx guaranteed to be less");
            let old_val = atomic.load(Ordering::Relaxed);
            if old_val == value {
                return Ok(InsertResult::AlreadyPresent);
            }
            if (old_val == 0 || old_val == TOMBSTONE)
                && atomic
                    .compare_exchange(old_val, value, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                return Ok(InsertResult::NotPresent);
            }
            attempt += 1;
        }
        Err(MapFull)
    }

    pub(crate) fn contains(&self, value: NonZeroU64) -> bool {
        self.idx(value).is_some()
    }

    fn idx(&self, value: NonZeroU64) -> Option<usize> {
        let value = value.get();
        let mut attempt = 0;
        while attempt < self.size() {
            let idx = self.hash(value, attempt);
            let atomic = self.els.get(idx).expect("idx guaranteed to be less");
            let stored_value = atomic.load(Ordering::Relaxed);
            match stored_value {
                0 => return None,
                v if v == value => return Some(idx),
                _ => attempt += 1,
            }
        }
        None
    }

    pub(crate) fn remove(&self, value: NonZeroU64) -> bool {
        if let Some(idx) = self.idx(value) {
            // if we've already removed that value, no worries
            let new_value = match value.get() {
                TOMBSTONE => 0,
                _ => TOMBSTONE,
            };
            self.els[idx]
                .compare_exchange(value.get(), new_value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use crate::tracked_spans::{InsertResult, TrackedSpans, TOMBSTONE};
    use std::collections::HashSet;
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    #[test]
    fn values_can_be_inserted() {
        let set = TrackedSpans::new(1024);
        assert!(!set.contains(nz(5)));
        set.insert(nz(5)).unwrap();
        assert!(set.contains(nz(5)));
        assert_eq!(set.insert(nz(5)), Ok(InsertResult::AlreadyPresent));
        assert_eq!(set.insert(nz(1234)), Ok(InsertResult::NotPresent));
        assert!(set.contains(nz(1234)));
    }

    #[test]
    fn map_can_fill_up() {
        // space for 3 items + TOMBSTONE
        let set = TrackedSpans::new(4);
        set.insert(nz(1)).unwrap();
        set.insert(nz(2)).unwrap();
        set.insert(nz(3)).unwrap();
        set.insert(nz(4)).expect_err("map full");
        set.insert(nz(TOMBSTONE)).expect("ok");
        set.insert(nz(1)).expect("ok, already there");

        set.remove(nz(1));
        set.insert(nz(1000)).expect("space now");
        assert!(set.contains(nz(1000)));
    }

    #[test]
    fn tombstone_can_be_inserted() {
        let set = TrackedSpans::new(1024);
        assert!(!set.contains(nz(TOMBSTONE)));

        set.insert(nz(TOMBSTONE)).unwrap();
        assert!(set.contains(nz(TOMBSTONE)));
        set.insert(nz(TOMBSTONE)).unwrap();
        assert!(set.contains(nz(TOMBSTONE)));
        assert!(set.remove(nz(TOMBSTONE)));
        assert!(!set.contains(nz(TOMBSTONE)));
    }

    #[test]
    #[cfg(loom)]
    fn test_concurrent_usage() {
        let collection_size = 3;
        tracing_subscriber::fmt::init();
        loom::model(move || {
            let tracked_spans = loom::sync::Arc::new(TrackedSpans::new(collection_size));
            let second_structure = loom::sync::Arc::new(loom::sync::RwLock::new(HashSet::new()));
            let mut threads = vec![];
            for t in 0..2 {
                let thread_copy = tracked_spans.clone();
                let map_copy = second_structure.clone();
                threads.push(loom::thread::spawn(move || {
                    let mut range: Box<dyn Iterator<Item = u64>> = Box::new(1..10);
                    if t % 2 == 0 {
                        range = Box::new((1..10).rev());
                    }
                    for i in range.take(2) {
                        if thread_copy.contains(nz(i)) {
                            assert!(map_copy.read().unwrap().contains(&i));
                        }
                        let mut guard = map_copy.write().unwrap();
                        guard.insert(i);
                        drop(guard);
                        if thread_copy.insert(nz(i)).is_ok() {
                            assert!(thread_copy.contains(nz(i)));
                        }
                    }
                }));
            }
            let thread_copy = tracked_spans.clone();
            let map_copy = second_structure.clone();
            threads.push(loom::thread::spawn(move || {
                for i in 1..5 {
                    if thread_copy.contains(nz(i)) {
                        assert!(map_copy.read().unwrap().contains(&i));
                    }
                }
            }));
            for handle in threads {
                handle.join().unwrap();
            }
            assert_eq!(
                (1..10).filter(|i| tracked_spans.contains(nz(*i))).count(),
                collection_size - 1
            );
        })
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_insertion(values in prop::collection::vec(1..u64::MAX, 1..100), checks in prop::collection::vec(1..u64::MAX, 1..1000)) {
            let sut = TrackedSpans::new(100);
            let mut check = HashSet::new();
            for v in values.iter() {
                sut.insert(NonZeroU64::new(*v).unwrap()).unwrap();
                check.insert(v);
            }
            for v in values.iter() {
                assert!(sut.contains(nz(*v)));
            }

            for v in checks.iter() {
                assert_eq!(sut.contains(nz(*v)), check.contains(v));
            }

            for v in values.iter() {
                let v = nz(*v);
                assert_eq!(sut.contains(v), check.contains(&v.get()));
                check.remove(&v.get());
                sut.remove(v);
                assert!(!sut.contains(v));
            }
        }
    }
}
