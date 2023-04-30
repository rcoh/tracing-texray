use std::fmt::{Debug, Formatter};
use std::num::NonZeroU64;

use std::sync::atomic::{AtomicU64, Ordering};

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

    fn size(&self) -> u64 {
        self.els.len() as u64
    }

    pub(crate) fn insert(&self, value: NonZeroU64) -> Option<bool> {
        assert_ne!(value.get(), TOMBSTONE);
        let value = value.get();
        let hash = value % self.size();
        let mut attempt = 0;
        while attempt < self.size() {
            let idx = ((hash + attempt) % self.size()) as usize;
            let atomic = self.els.get(idx).expect("idx guaranteed to be less");
            let old_val = atomic.load(Ordering::Relaxed);
            if old_val == value {
                return Some(false);
            }
            if (old_val == 0 || old_val == TOMBSTONE) && atomic
                    .compare_exchange(old_val, value, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok() {
                return Some(true);
            }
            attempt += 1;
        }
        None
    }

    pub(crate) fn contains(&self, value: NonZeroU64) -> bool {
        self.idx(value).is_some()
    }

    fn idx(&self, value: NonZeroU64) -> Option<usize> {
        let value = value.get();
        let hash = value % self.size();
        let mut attempt = 0;
        while attempt < self.size() {
            let idx = ((hash + attempt) % self.size()) as usize;
            let atomic = self.els.get(idx).expect("idx guaranteed to be less");
            let stored_value = atomic.load(Ordering::SeqCst);
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
            self.els[idx]
                .compare_exchange(value.get(), TOMBSTONE, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hashset::TrackedSpans;
    use std::collections::HashSet;
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    #[test]
    fn values_can_be_inserted() {
        let set = TrackedSpans::new(1024);
        assert_eq!(set.contains(nz(5)), false);
        set.insert(nz(5));
        assert_eq!(set.contains(nz(5)), true);
        assert_eq!(set.insert(nz(5)), Some(false));
        assert_eq!(set.insert(nz(1234)), Some(true));
        assert_eq!(set.contains(nz(1234)), true);
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn test_insertion(values in prop::collection::vec(1..100000u64, 1..100), checks in prop::collection::vec(1..10000u64, 1..1000)) {
            let sut = TrackedSpans::new(100);
            let mut check = HashSet::new();
            for v in values.iter() {
                assert_eq!(sut.insert(NonZeroU64::new(*v).unwrap()), Some(!check.contains(v)));
                check.insert(v);
            }
            for v in values.iter() {
                assert_eq!(sut.contains(nz(*v)), true);
            }

            for v in checks.iter() {
                assert_eq!(sut.contains(nz(*v)), check.contains(v));
            }

            for v in values.iter() {
                let v = nz(*v);
                assert_eq!(sut.contains(v), check.contains(&v.get()));
                check.remove(&v.get());
                sut.remove(v);
                assert_eq!(sut.contains(v), false);
            }
        }
    }
}
