/** The unit table, shared by the board and the move record.
 *
 * One copy, because there were two and they disagreed: the record's private
 * copy had an extra unit in it and named every index above zombie wrongly.
 * The layout is already hand-duplicated across the Rust crate, the learner and
 * the record format, so the web side at least gets to have a single source.
 */

export interface UnitDef {
  name: string;
  ch: string;
  attack: string;
  damage: number; // 0 for unsummon
  defense: number;
  speed: number;
  range: number;
  cost: number;
  rebate: number;
  necromancer?: boolean;
  lumbering?: boolean;
  flying?: boolean;
  persistent?: boolean;
  spawn?: boolean;
}

/** Index = unit byte in the payload. Must match crates/games/minions/src/units.rs. */
export const UNITS: UnitDef[] = [
  { name: "necromancer", ch: "N", attack: "unsummon", damage: 0, defense: 7, speed: 1, range: 1, cost: 0, rebate: 0, necromancer: true, persistent: true, spawn: true },
  { name: "zombie", ch: "Z", attack: "1", damage: 1, defense: 1, speed: 1, range: 1, cost: 2, rebate: 0, lumbering: true, spawn: true },
  { name: "skeleton", ch: "S", attack: "5", damage: 5, defense: 2, speed: 1, range: 1, cost: 4, rebate: 2, flying: true, persistent: true },
  { name: "serpent", ch: "P", attack: "3", damage: 3, defense: 1, speed: 2, range: 1, cost: 4, rebate: 2 },
  { name: "warg", ch: "W", attack: "1", damage: 1, defense: 1, speed: 3, range: 1, cost: 4, rebate: 2 },
  { name: "ghost", ch: "G", attack: "1", damage: 1, defense: 4, speed: 1, range: 1, cost: 3, rebate: 1, flying: true, persistent: true },
];
/** Buyable units, as unit indices (everything but the necromancer). */
export const BUYABLE = [1, 2, 3, 4, 5];
