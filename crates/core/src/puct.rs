//! Shared PUCT details that every search in the repo has to agree on.
//!
//! Right now that is **first-play urgency**: the value assumed for a child the
//! search has not visited yet.
//!
//! The obvious choice, `q = 0`, is a trap. It is only neutral if the value head
//! is centred at zero, and a net that is merely *biased* — early in training,
//! say — breaks the search in two opposite ways:
//!
//!   * if some child evaluates strongly in the mover's favour, its `q` beats
//!     every unvisited child's 0 forever and the search tunnels down one line
//!     (measured: 30 of 30 root visits on a single move);
//!   * if every child evaluates against the mover, every *visited* child sinks
//!     below the unvisited ones' 0, so the search spreads exactly one visit per
//!     move and the move played is decided by the argmax tie-break rather than
//!     by search.
//!
//! Which regime a position falls into correlates with whose turn it is, so the
//! damage is not even symmetric: in Hex self-play this handed one colour every
//! single game (0 of 320) while the same net under a search with a different
//! tie-break handed them all to the other colour.
//!
//! The fix is the standard one (AlphaZero, Leela, KataGo): treat an unvisited
//! child as worth roughly what its parent is worth, discounted a little by how
//! much prior mass has already been tried. That keeps the comparison between
//! visited and unvisited children on the same scale whatever the value head's
//! bias, and it reduces to the old behaviour on the first selection at a node
//! (nothing visited ⇒ no reduction, and an unvisited parent's q is 0), so the
//! opening choice at each node is still made by the policy prior alone.

/// How much to discount an unvisited child once the whole prior mass has been
/// tried. Leela/KataGo territory (0.2–0.25); larger favours exploitation of
/// what has already been searched.
pub const FPU_REDUCTION: f32 = 0.2;

/// Value to score an unvisited child with.
///
/// * `parent_q` — the node's own mean value, from *its* mover's point of view
///   (i.e. `Σw / Σn` over its edges, since backup already negates per ply). 0
///   when the node has no visits yet.
/// * `visited_prior_mass` — summed prior of the children visited so far, in
///   [0, 1].
#[inline]
pub fn fpu(parent_q: f32, visited_prior_mass: f32) -> f32 {
    parent_q - FPU_REDUCTION * visited_prior_mass.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_selection_at_a_node_is_unchanged() {
        // Nothing visited, parent unvisited: exactly the old `q = 0`, so the
        // first child is still chosen by prior alone.
        assert_eq!(fpu(0.0, 0.0), 0.0);
    }

    #[test]
    fn unvisited_children_track_the_parent_rather_than_zero() {
        // A node the search believes is lost: unvisited children are *also*
        // roughly lost, not a free 0 that outranks everything explored.
        assert!(fpu(-0.8, 0.5) < -0.8);
        assert!(fpu(-0.8, 0.5) > -1.2, "the reduction stays small");
        // And in a won node they don't look worse than the explored ones by
        // more than the reduction.
        assert!(fpu(0.8, 1.0) > 0.8 - 0.3);
    }

    #[test]
    fn reduction_grows_with_the_prior_mass_already_tried() {
        let none = fpu(0.0, 0.0);
        let some = fpu(0.0, 0.25);
        let all = fpu(0.0, 1.0);
        assert!(all < some && some < none);
        assert_eq!(all, -FPU_REDUCTION);
    }
}
