//! A small multiplicative hasher for internal, integer-keyed maps.
//!
//! `std::collections::HashMap`'s default `RandomState` uses SipHash, which is
//! designed to resist hash-flooding attacks on attacker-controlled keys. The
//! hot maps in this crate are keyed on dense integers this crate's own
//! compiler assigns (NFA/DFA state ids) or on small derived structures built
//! from them (`StateKey`), never on external input, so there is nothing here
//! for SipHash's per-byte mixing and finalization to defend against — it is
//! pure overhead.
//!
//! [`FxHasher`] is deterministic by construction: the same input always
//! produces the same hash, across runs and across processes. That is
//! intentional and safe *because* the key space is internal and
//! non-adversarial — do not reach for this hasher for anything keyed on
//! external or attacker-influenced data (e.g. pattern text, named capture
//! group names), where SipHash's randomized keying is a real defense, not
//! overhead.

use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative seed, chosen for good bit dispersion (same constant used by
/// rustc's internal FxHash and the `rustc-hash` crate).
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// A fast, non-cryptographic hasher for small integer keys.
///
/// See the module docs for why determinism and the lack of DoS-resistance
/// are acceptable here.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        // Bulk path for types that hash raw bytes rather than going through the
        // `write_u*` methods. `StateKey` derives `Hash` over an `Arc<[u32]>`
        // and a `CharClass` enum, and slices of integers hash their elements as
        // one byte run, so this is where the DFA state map's keys are actually
        // mixed — two state ids per `add` instead of one.
        while bytes.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            self.add(u64::from_ne_bytes(buf));
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut tail = 0u64;
            for (i, &b) in bytes.iter().enumerate() {
                tail |= (b as u64) << (i * 8);
            }
            self.add(tail);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `BuildHasher` for [`FxHasher`], for use as a `HashMap`/`HashSet` type
/// parameter.
pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// A `HashMap` keyed by dense internal integers (or structures built from
/// them), hashed with [`FxHasher`] instead of the default SipHash.
pub(crate) type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

/// A `HashSet` over dense internal integers (or structures built from them),
/// hashed with [`FxHasher`] instead of the default SipHash.
pub(crate) type FxHashSet<K> = std::collections::HashSet<K, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = FxHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn same_input_hashes_equal() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(hash_of(&42u32), hash_of(&42u32));
        assert_eq!(hash_of(&42u64), hash_of(&42u64));
        assert_eq!(hash_of(&bytes.as_slice()), hash_of(&bytes.as_slice()));
    }

    #[test]
    fn different_inputs_hash_differently() {
        // Not a formal guarantee for arbitrary pairs, but these are the exact
        // shapes of key this hasher exists to serve, and they must not
        // collide in practice.
        assert_ne!(hash_of(&1u32), hash_of(&2u32));
        assert_ne!(hash_of(&1u64), hash_of(&2u64));
        assert_ne!(hash_of(&0u32), hash_of(&u32::MAX));
        assert_ne!(
            hash_of(&b"short".as_slice()),
            hash_of(&b"a-different-longer-slice".as_slice())
        );
    }

    #[test]
    fn hashmap_round_trips_with_collisions_by_construction() {
        // Keys chosen so several pairs share `key % 8` (FxHasher's low bits
        // before the final table has a chance to redistribute), forcing the
        // map to resolve real bucket collisions rather than only ever seeing
        // spread-out keys.
        let keys: Vec<u32> = (0u32..64).chain([1000, 1008, 1016, 2024]).collect();

        let mut map: FxHashMap<u32, String> = FxHashMap::default();
        for &k in &keys {
            map.insert(k, format!("value-{k}"));
        }

        for &k in &keys {
            assert_eq!(map.get(&k), Some(&format!("value-{k}")));
        }
        assert_eq!(map.len(), keys.len());
        assert_eq!(map.get(&999), None);
    }
}
