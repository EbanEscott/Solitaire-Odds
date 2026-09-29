# The Complexity of Solitaire

- **Citation key:** `longpre2009complexity`
- **Authors:** Luc Longpré; Pierre McKenzie
- **Publication:** Theoretical Computer Science 410(50), 5252–5260 (2009).
- **Local PDF:** [The Complexity of Solitaire](the-complexity-of-solitaire.pdf)
- **Source and version:** [University of Campinas teaching copy](https://ic.unicamp.br/~santiago/assets/mc558/2024%20-%202%20-%20projects/the%20complexity%20of%20solitaire.pdf); journal article, [DOI](https://doi.org/10.1016/j.tcs.2009.08.027).
- **Downloaded:** 2026-09-29
- **PDF pages:** 9
- **Related links:** [Publisher](https://www.sciencedirect.com/science/article/pii/S0304397509006100) · [DOI](https://doi.org/10.1016/j.tcs.2009.08.027)

## Summary

Analyzes the computational complexity of deciding whether a generalized Klondike configuration has a winning sequence.

## Key findings

- Proves NP-completeness for the generalized n-card problem; see the abstract and formal results in the paper.

## Conditions and limitations

The asymptotic result concerns growing instances. It is not a complexity classification of one fixed 52-card deck and provides neither a human win rate nor a hidden-information policy benchmark.

## Relevance to Solitaire Odds

Our interpretation: theoretical motivation for computational search and structural deadlock analysis, rather than an empirical player baseline.

## Extracted PDF text

Apple Vision OCR, extracted 2026-09-29. Page headings below use physical PDF page numbers, including covers and front matter; printed page numbers can differ. OCR was used because the embedded text loses spaces between words. Mathematical notation and symbols can be misrecognized; consult the PDF for exact statements. Unmapped control characters are shown as `�`. The text below is source material, separate from our summary and interpretation above.

### PDF page 1

```text
Theoretical Computer Science 410 (2009) 5252-5260
ELSEVIER
Contents lists available at ScienceDirect
Theoretical Computer Science
journal homepage: www.elsevier.com/locate/tcs
Theoretical
Computer Scienc
The complexity of Solitaire
Luc Longpré?, Pierre McKenzie b,*
a Computer Science, University of Texas at El Paso, United States
• DIRO, Université de Montréal, Canada
ARTICLE INFO
Keywords:
Computational complexity
Completeness
Games
Solitaire
ABSTRACT
Klondike is the well-known 52-card Solitaire game available on almost every computer.
The problem of determining whether an n-card Klondike initial configuration can lead to a
win is shown NP-complete. The problem remains NP-complete when only three suits are
allowed instead of the usual four. When only two suits of opposite color are available, the
problem is shown NL-hard. When the only two suits have the same color, two restrictions
are shown in AC® and in NL respectively. When a single suit is allowed, the problem drops in
complexity down to AC [3], that is, the problem is solvable by a family of constant-depth
unbounded-fan-in {AND, OR, mod3 } -circuits. Other cases are studied: for example,
"no
King" variant with an arbitrary number of suits of the same color and with an empty "pile"
is NL-complete.
© 2009 Elsevier B.V. All rights reserved.
1. Introduction
Solitaire card games, called patience games outside of the United States, apparently originate from the fortune-telling
circles of the eighteenth century [9]. Of the many hundred different solitaire card games in existence [8], to the best of our
knowledge, only FreeCell [4] and BlackHole [5] have been studied from a complexity viewpoint. In both cases, determining
whether an initial configuration can lead to a win was shown NP-complete.
Over the last two decades, a particular variation of solitaire, the Klondike version, was popularized by Microsoft Windows
(Fig. 1). Earlier, Parlett [8] had described Klondike as the "most popular of all perennial favorites in the realm of patience,
which is surprising since it offers the lowest success rate of any patience".
In a paper entitled Solitaire: Man Versus Machine, Yan, Diaconis, Rusmevichientong and Van Roy [10] report that a human
expert (and distinguished combinatorialist, former president of the American Mathematical Society!) patiently recorded data
on his playing 2000 games of thoughtful solitaire, that is, the intellectually more challenging Klondike in which the complete
initial game configuration is revealed to the player at the start of the game. The expert averaged 20 min per game and was
able to win the game 36.6% of the time. Pointing to the prohibitive difficulty of obtaining nontrivial mathematical estimates
on the odds of winning this common game, Yan et al. then describe heuristics developed using the so-called rollout method,
and they report a 70% win rate.
Why is it so hard to compute the odds of winning at Klondike? In part, this question prompted our investigation of
the complexity of the game. As well, after using the Minesweeper game [6] for several years as a motivating example for
students, we looked for different NP-complete examples based on other widely popular and deceptively simple games.
The precise rules of Klondike are described in Section 2. To make the game amenable to a computational complexity
analysis, the game is generalized to allow instances of arbitrary finite size. Hence an instance of the game involves a "deck"
containing the "cards"
* Corresponding author.
E-mail addresses: longpre@utep.edu (L. Longpré), mckenzie@iro.umontreal.ca (P. McKenzie).
0304-3975/$ - see front matter © 2009 Elsevier B.V. All rights reserved.
doi: 10.1016/j.tcs.2009.08.027
```

### PDF page 2

```text
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
5253
• Solitaire
Game Help
Score: -$52
Fig. 1. Initial Klondike configuration on Microsoft Windows. In the terminology of [10], there are four empty suit stacks, seven build stacks containing 1, 2,
3, 4, 5, 6 and 7 cards respectively with only the top card facing up, and one pile containing the remaining cards facing down; another stack, the talon, will
appear in the course of the game when cards from the pile are moved to the talon three by three.
16, 2eo, 3do,
...n - 16o, nelo
14,24, 34,..., п - 18, п
10,20, 30,..., п - 10, по
100, 200, 30, ..., п - 100, по
and the game configurations are generalized appropriately. The problem of interest is to determine, given an initial game
configuration, whether the game can be won. We show here that this is NP-complete.
Our NP-completeness proof bears resemblance to the proof that FreeCell is NP-complete [4]. In particular, the method of
nondeterministically assigning truth values to card configurations is the same. However, our strategy gets by with only three
card suits as opposed to the usual four used in the current proof that FreeCell is NP-complete. Many differences between
FreeCell and Klondike further arise, for instance, when we argue the NP upper bound in the presence of "backward" moves
(i.e., from a suit stack to a build stack) and when we consider restricted variants of the game. We highlight the following as
our main results:
1. Klondike is NP-complete and remains so with only three suits available,
2. Klondike with a black suit and a red suit is NL-hard,
3. Klondike with any fixed number b of black suits is in NL,
4. flat Klondike (that is, without a pile) with an input-dependent number of black suits and without generalized Kings (see
below) is NL-complete,
5. Klondike with a single suit is in AC 13],
6. flat Klondike with 2 black suits and without generalized Kings is in AC®.
Section 2 contains preliminaries and a precise description of the Klondike variation studied in this paper. Section 3
proves that Klondike is NP-complete and Section 4 considers restricting the usual four-suit game. Section 5 concludes with
a discussion and some open questions.
2. Preliminaries
2.1. Complexity theory
We assume familiarity with basic complexity theory, such as can be found in standard books on the subject, for example
[7]. We recall the inclusion chain
AC° C AC [3] C AC° [6] E L E NL = CO-NL E P E NP.
Here, the complexity class AC is the set of languages accepted by DLOGTIME-uniform unbounded-fan-in constant
depth {^, V, -}-circuits of polynomial size. The larger class AC® [m] is the set of languages AC®-Turing reducible to the
MODm Boolean function, defined to output 1 iff m does not divide the sum of its Boolean inputs. The classes L and NL
stand for deterministic and nondeterministic logarithmic space respectively. The classes P and NP are deterministic and
```

### PDF page 3

```text
5254
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
nondeterministic polynomial time respectively. We adopt the definitions of [3] for constant-depth circuit uniformity (see
also [2]).
If not otherwise stated, the hardness results in this paper are in terms of many-one logspace reducibility.
2.2. Klondike
We allow ourselves to borrow the following excellent description of Klondike found in [10, Section 2):
he goal of the game is to move all cards into the suit stacks, aces first, then two's, and so on, with each suit stack
volving as an ordered increasing arrangement of cards of the same suit. On each turn, the player can move cards
from one stack to another in the following manner:
1. Face-up cards of a build stack, called a card block, can be moved to the top of another build stack provided that
the build stack to which the block is being moved accepts the block (see Points 6 and 7 below for the meaning of
acceptance). Note that all face-up cards on the source stack must be moved together. After the move, these cards
would then become the top cards of the stack to which they are moved, and their ordering is preserved. The card
originally immediately beneath the card block, now the top card in its stack, is turned face-up. In the event that all
cards in the source stack are moved, the player has an empty stack.
2. The top face-up card of a build stack can be moved to the top of a suit stack, provided that the suit stack accepts
3. The top card of a suit stack can be moved to the top of a build stack, provided that the build stack accepts the card.
4. If the pile is not empty, a move can deal its top three cards to the talon, which maintains its cards in a first-in-last-
out order. If the pile becomes empty, the player can redeal all the cards on the talon back to the pile in one card
move. A redeal preserves the ordering of cards. The game allows an unlimited number of redeals.
5. A card on the top of the talon can be moved to the top of a build stack or a suit stack, provided that the stack to
which the card is being moved accepts the card.
a card block whose bottom card is a King.
7. A suit stack can only accept an incoming card of its corresponding suit. If a suit stack is empty, it can only accept
an Ace. If it is not empty, the incoming card must be adjacent to the current top card of the suit stack.
Yan et al. coin the name thoughtful solitaire for the Klondike variation in which the player sees the complete game
configuration, including the ranks of all the cards facing down, throughout the course of the game. The Klondike rules are
Since we generalize Klondike to involve a variable number of cards, we adjust the notion of a game configuration to allow
an arbitrary initial number of build stacks of arbitrary size. No new build stacks can be created in the course of the game
however. We do not insist that the initial build stacks contain k, k - 1,..., 2, 1 cards respectively, but this could be arranged
in most cases with no loss of generality by shifting all the card values upward and then inflating the initial build stacks with
low value cards that can be released right away. Note that because the creation of new build stacks that may become empty
quickly would allow the movement of generalized Kings, this method does not provide a general transformation from an
instance with an arbitrary set of stacks. However, since none of our lower bound proofs use properties of generalized Kings,
they all hold with this additional restriction. We will occasionally consider Klondike with a number of suits other than 4. In
that case, the number of suit stacks is adjusted accordingly.
Card numbers and suit numbers are represented in binary notation. We assume any reasonable encoding of cards and
game configurations that allows extracting individual card information in AC°. In particular, the pile, talon and stacks are
represented in table form so that the predicate "c is the ith card in the table" is AC'-computable.
Definition 1. Problem SoLIT(b, r):
Given: an initial b-black-suit and r-red-suit Klondike configuration involving the same number n of cards in every suit.
Determine: Whether the (b + r)n cards can be placed on the b + r suit stacks by applying the Klondike game rules starting
from the given initial configuration.
In Section 3 we will be studying SolitaIre, by which we mean SoLIt(2, 2). In Section 4, we will consider Klondike
restrictions, such as FLAT-SOLIT(b, r), by which we mean Solit(b, r) with an initial configuration having an empty pile
and empty talon. We define the further restriction FLAT-SOLITNoKing (b, r) to mean FLAT-SOLIT(b, r) played with modified
rules that forbid an empty stack from accepting a generalized King (i.e., we disallow refilling an empty build stack; this is
equivalent to viewing the highest ranked cards as generalized Queens rather than generalized Kings). Using a "*", such as
in FLAT-SOLIT(*, O), means that the number of suits corresponding to the * is not fixed and depends on the input.
3. Klondike is NP-complete
Theorem 2. SOLITAIRE is NP-complete.
```

### PDF page 4

```text
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
5255
Proof (NP Upper Bound). Consider a winning N-card Klondike instance involving k build stacks. We need to argue that the
length of a shortest winning sequence of moves is polynomial in N. We define four types of moves:
Type 1: moving a card out of the talon, or any move that causes a face-down card in a build stack to turn face-up. This
includes any move from build stack to build stack, or moving the last face-up card from a build stack to a suit stack
(except when such a move empties a build stack).
Type 2: moving a card from a build stack to a suit stack (without being of type 1).
Type 3: moving a card from a suit stack to a build stack.
Type 4: moving cards from the pile to the talon.
Let l be a shortest winning sequence of moves. Such a sequence contains N — k moves of type 1 since exactly k cards
were visible at the outset. We claim that two successive moves of type 1 in this sequence are separated by O(N) moves of
type 2, type 3 or type 4. The NP upper bound follows from the claim since moves of type 1 are irreversible and no obstacle
remains after the N — k moves of type 1; thus l is O(N2).
To see the claim, we show that if there is a winning sequence, then there is one where the moves between two successive
moves of type 1 consist of a sequence of at most N moves of type 2, followed by at most N moves of type 3, followed by at
most N moves of type 4. First, moves of type 2 and of type 3 do not interfere with moves of type 4, so we can delay all type
4 moves until all type 2 and type 3 moves are finished.
Now, it remains to see that type 3 moves can be assumed to occur last in an optimal sequence of moves of types 2 or 3.
This will suffice since there can be no more than N consecutive type 2 moves and no more than N consecutive type 3 moves.
So consider a configuration C and an optimal sequence 8 of moves of types 2 or 3 leading to a configuration C'. We prove,
by induction on the number k of type 3 moves, that postponing the type 3 moves in s until the end still leads from C to C'. If
k = 0 then this is vacuously true. So let k > O. Let 8, the first type 3 move in S, remove a card c from a suit stack 0c. Let C1 be
the configuration reached from C by applying the prefix of S up to and including 8. Since the remainder of S leads optimally
from Cy to C' in k - 1 type 3 moves, by the induction hypothesis, these k - 1 type 3 moves can be postponed until after the
type 2 moves. Let S' be the sequence, leading optimally from C to C', obtained from S by postponing its last k — 1 type 3
moves until the end. Note that no type 2 move involving o, can occur after & in S' until c is moved back to oc. But moving c
back to o, after 8 would merely undo 8, contradicting the optimality of s'. Consequently, because no type 2 move after 8 in
s' involves oc, 8 can also be postponed until after all type 2 moves in s'. This concludes the induction. Hence, between any
two successive type 1 moves in a shortest sequence, there can be at most 3N moves of types 2, 3 or 4, proving the claim.
NP-hardness. We reduce from 3SAT. The main idea is to construct a pair of build stacks for each 3SAT formula variable.
The top cards on these stacks will correspond to whether we want the variable to be true or false. Only one of these two top
cards can be moved to another build stack. Moving a top card will uncover cards corresponding to the clauses that become
true when the variable takes the Boolean value associated with the top card.
Let F be a 3CNF Boolean formula with n variables v1, ..., Un and m clauses c1, ..., Cm. Construct an initial configuration
corresponding to this formula so that the configuration is winning if and only if the formula is satisfiable. For each variable
Vi, associate cards 2i et 2i + 1. For each clause C;, associate cards 2n + 7j,..., 2n + 7j + 6.
For each variable vi, construct 3 build stacks (called the variable stacks):
1. one with card 2ic facing up on top and cards to be determined below,
2. one with card 2ieo facing up on top and cards to be determined below,
3. one with the sole card (2i + 1)C.
For each clause c; = (Ip, lq, 4r), construct 3 build stacks (called clause stacks) with one card facing up on top and one card
facing down below:
1. one stack with (2n + 7j + 6)© on top and (2n + 7j + 5)d below,
2. one stack with (2n + 7j + 4)C on top and (2n + 7j + 3)d below,
3. one stack with (2n + 7j + 2) on top and (2n + 7j + 1)& below.
BETE
Also, if lp = Vi, put (2n + 7j + 1) A facing down in stack 2it, and if lp = Vi, put (2n + 7j + 1) facing down in stack 2is.
If lq = Vi, put (2n + 7j + 3)A facing down in stack 2ie, and if lą = Vi, put (2n + 7j + 3) facing down in stack 2ioo. If ly = Vi,
put (2n + 7j + 5)0 facing down in stack 2i, and if ly = Vi, put (2n + 7j + 5)% facing down in stack Zie. Arrange for all
clause (spade, face-down) cards within any given build stack to occur in order of increasing card rank.
Finally, create the critical build stack, facing down, with cards (2n + 7j) in any order for 1 ≤ j ≤ m, followed by
all remaining cards in increasing order, starting with aces, and followed further by 3 generalized Kings 2n + 7m + 70,
2n + 7m + 7% and 2n + 7m + 70. Regrouping all the generalized Kings at the bottom of the critical stack serves to prevent
moves to an empty build stack during the core of the simulation. The pile, talon and suit stacks are empty. For each clause
j, we will refer to the card 2n + 7j8 as to the critical clause-j card.
Suppose that some assignment satisfies the formula. Here is how to win the game. In the assignment, if the variable v¡
is false, put card 2ido on card (2i † 1)O. If v¡ is true, put card 2ig instead. Then, move all the cards that were below the 2i
cards. This is possible because all these cards are spade cards numbered 2n + 7j + 1 or 2n + 7j + 3 or 2n + 7j + 5, and the
red cards 2n + 7j + 2 and 2n + 7j + 4 and 2n + 7j + 6 all sit facing up on top of their build stacks. If the formula is satisfiable,
all the clauses have at least one literal set to true, so at least one clause j card will be released in this manner for each j.
Claim: For each j, a sequence of j-clause stack moves now exists such that
```

### PDF page 5

```text
5256
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
1. one clause-j stack can be made to accept the critical clause-j card, and
2. after this sequence, if a situation is reached such that all the cards ranked less than 2n + 7j are placed on the suit stacks
and a black card ranked 2n † 7j sits on top of the critical stack, then all the cards ranked 2n † 7j, 2n + 7j+ 1, ..., 2n+7j+6
can be placed on the suit stacks.
This claim implies a win as follows. Part (1) of the claim ensures that all the critical cards can be moved from the critical
stack to the clause stacks. This releases the aces and allows moving all the cards ranked less that 2n + 7, including those
that remained on the variable stacks, to the suit stacks. Part (2) of the claim together with an induction on j then yield the
winning sequence of moves.
To prove the claim, fix j and let 2n + 7j + ko for some k e {1, 3, 5} be the smallest clause-j card that got released from
a variable build stack. If k = 1, then 2n + 7j + koo was accepted by 2n + 7j + 20 and the resulting stack accepts the critical
clause-j card. If k = 3, then 2n † 7j + koo was accepted by 2n † 7j + 48, so the 2n + 7j † 29 card can be displaced, thus
uncovering 2n + 7j + 166 which in turn accepts the critical clause-j card. Finally, it k = 5, then 2n + 7j + ke was accepted
by 2n + 7j + 60; now 2n + 7j + 40 can be displaced, followed by 2n + 7j + 3&, followed by 2n + 7j + 20, again uncovering
2n + 7j + 1% which accepts the critical clause-j card. This proves part (1) of the claim. To prove part (2) of the claim, it
suffices to observe that although some 2n + 7j + kob card(s) remain(s) under some 2n + 7j + k + 10 card(s) after the above
sequence of moves, the resulting stack configurations do not form an obstacle when the complementary cards in all suits
are available in increasing order.
Conversely, suppose that the configuration produced from the formula is winning. Then the initial sequence of a winning
sequence of moves must uncover the aces. This initial sequence cannot involve backward moves (i.e., from a suit stack to a
build stack). This initial sequence must then first release every critical card 2n + 7jO. Each of these cards must be moved to
a black (2n + 7J+ 1) card. For any given j, this cannot happen unless for some i, some clause-j card is released from one (and
only one, since a single card 2i + 1 is visible) of the two vi-variable stacks. An assignment of variables v; based on which of
the two vj-variable stacks was first released is, by construction, a satisfying assignment to our formula. •
4. Complexity of Klondike restrictions
The proof of Theorem 2 used only (o, S, 0). Furthermore, the initial configuration constructed had an empty pile and
empty talon. Thus we have:
Theorem 3. SoLIT(2, 1) and FLAT-SOLIT(2, 1) are NP-complete.
Because the NP upper bound argument from Theorem 2 extends to the case in which an arbitrary number of (red and
black) suits is allowed, we also have:
Theorem 4. SoLIT (*, *) and FLAT-SOLIT (*, *) are NP-complete.
Recall the "no King" game restriction, in which empty build stacks can never be filled. Because the Klondike instances
constructed in the NP-hardness proof from Theorem 2 neither allow nor tolerate refilling an empty stack (except at the very
end when all the cards have been released), we also have:
Theorem 5. FLAT-SOLITNoKing (b, r) is NP-complete for any b > r ≥ 1.
One might expect the remaining cases, namely the case of one black suit and one red suit, and the case in which all suits
are black, to be trivial. This is not quite so. We begin with the latter.
A FLAT-SOLIT(*, O) instance w involves a set of nb cards C1,1, C1,2, ... , C1,b, C2,1,..., ...,•.., Cn, scattered within an arbitrary
number of build stacks, where the card Cis is the suit-s card of rank i. Since only black suits occur in w, the only actions
possible are those that move a generalized King and its block to an empty build stack and those that move a card from a
build stack to a suit stack. Even when all suits are black, the generalized King moves are powerful because the choice of
which black King to move to an empty stack can be critical to the successful completion of the game. We do not yet fully
understand the power of such moves. So we turn to FLAT-SOLITNoking (*, O).
We say that a FLAT-SOLITNoKing (*, O) instance w is nontrivial if for every s, the suit-s cards occur in increasing order in
every build stack. Clearly, no win is possible from a trivial w. When w is nontrivial, we define the directed graph H(w) on
the set of cards of w as follows: for 1 ≤ i, j ≤ nand 1 ≤ s, t ≤ b, the arc (Ci,s, Cj,t) exists in H(w) iff
1. s = t and j = i - 1 (call this a horizontal edge), or
2.5 # t and the card Ci,s is immediately beneath Cj,t in some build stack (call this a vertical edge).
Proposition 6. Consider a nontrivial FLAT-SOLITNoking (*, O) instance w. A win is possible from w iff H(w) is cycle-free.
Proof. Suppose that a cycle exists in H(w). By construction, such a cycle must involve at least one horizontal edge
(Ci,s, Ci-1,s). Since H (w) obviously captures implication chains of the form "a card c cannot be placed on a suit stack before
all the cards reachable from c in H(w) are themselves placed on a suit stack", this cycle implies that cis must be placed on
a suit stack before Ci-1,s. Hence a win is impossible from w.
Conversely, suppose that H is cycle-free. Then there exists an (inverse) topological sort of H(w), that is, an ordering
,.... con) of the nodes of H (w) such that no edge (c *), c) with k < 1 appears in H(w). We prove by induction on
```

### PDF page 6

```text
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
5257
k that if c('), c(2)
., C(k-1) are on the suit stack, then c(k) can be placed on a suit stack.
Basis: If k = 1, we know that no edge out of c'" appears in H(w). By construction, c() must be C1,s for some suit s and must
appear on top of a build stack. Hence c() can form a (first) suit stack.
Inductive step: Let k > 1 and suppose now that c(1)
,.... c(k-1) are on the suit stacks. If c(k) = Ci,s cannot be moved to
a suit stack, then either Cis is immediately beneath some Ct not yet on the suit stacks, or Ci-1,s is not yet on the suit stacks.
In both cases, because the cards not yet on a suit stack occur as c for some I > k, an edge (cl), c() with k < I must occur
by construction of H(w). But this contradicts the properties of the topological ordering. This completes the induction and
proves that all the cards can be moved to the suit stacks. Hence a win is possible from w. •
Theorem 7. FLAT-SOLITNoKing (*, O) is NL-complete.
Proof. NL upper bound. Consider a FLAT-SOLITNoKing (*, 0) instance w. We first check in AC° that w is nontrivial. If w is trivial
then we reject immediately. Otherwise, Proposition 6 implies a co-NL = NLupper bound, because H (w) is easily constructed
in log space (in AC® in fact), and the total number bn of nodes in H(w) together with the card numbers and suit numbers
involved in w are O(log n)-bit numbers.
NL-hardness. We reduce to FLAT-SOLITNoKing (*, 0) the co-NL-complete problem of determining whether no path exists
trom node s to node t Fs in a directed graph G without selt-loops and with edge set E § 11,...,n, × 11,...,ng. The
reduction is rendered delicate by the fact that several cards need to be assigned to each node in G: this is because a card can
only occur once in a Klondike instance and furthermore, the "horizontal requirements" arising from the ranks of the cards
are of course not compatible with the implicit ordering of the nodes in G. We now describe the flat Klondike instance w
produced from G. It uses the cards Ciu, 1 ≤ i ≤ 2n', 1 ≤ u ≤ n, arranged in |E|(n - 1) + 2 build stacks as follows:
1. for each edge (i, j) in E and for each k, 0 ≤ k < n - 1, one stack with Ck(2n) tij on top and with C(k+ 1) (2n) +n+j,; facing down
2. one stack with C2n2,s on top and with C1,t facing down below
3. one stack with the remaining cards facing down, in increasing order.
The reader can check that no card is mentioned twice in this (log space) construction. Furthermore, w is nontrivial, since
only one build stack contains two cards of the same suit, and this stack is properly ordered. Hence the graph H (w) is defined.
It is a n × 2n grid with the horizontal edges forming n parallel lines running from left to right (card ranks decrease from left
To see how the vertical edges operate, imagine the grid partitioned into columns n, n - 1,..., 1 of width 2n. Each such
(i, j) € E run from the source region in every column k + 1 on line i to the target region in column k on line j. Observe
then that the vertical H (w) edges arising from E together with the horizontal H (w) edges are incapable of forming a cycle
in H (W). The vertical edge (C1,t, C2n2,
) is the only edge in H (w) which connects a column (in fact, the rightmost entry in the
source region of the nth column on line t) to a column situated to its left (in fact, to the leftmost entrv in the target region
of the first column on line s).
It follows that if a cycle exists in H(w), then (C1,t, C2n2,s) is part of it. Hence, if such a cycle exists, a path exists in H(w)
from the line s to the line t. This implies that a path existed from s to t in G.
Conversely, if a path s = v1, 02, ..., Um = t with m ≤ n - 1 exists in G, then a path can be traced from the column n on
line s to the column n — m on line t in H(w) by appropriately combining neighbouring column traversals with horizontal
displacements on the successive lines v1, U2, ..., Um. A final horizontal displacement leads to C1,t and thus to C2n2,s, creating
a cycle in H (W).
Hence a cycle exists in H (w) iff a path exists in G. By Proposition 6, a path exists in G iff no win is possible from w. This
concludes the NL-hardness proof. •
We now relate the case of an arbitrary number of black suits to the case of a red suit and a black suit. We can show that
when generalized King moves are disallowed, the all-blacks case reduces to the case of a red suit and a black suit.
Proposition 8. FLAT-SOLITNoKing (*, O) AC -reduces to FLAT-SOLITNoKing (1, 1).
Proof. Let a FLAT-SOLITNoKing (*, O) instance w involve the nb cards C1,0, C1,1, . . ., C1,b-1, C2,0, ...
., Cn, b-1 where Ci,s is the
suit-(s + 1) card of rank i. The idea is to rename each ci,s as a 8 card, and to use of cards to restrict the release of the renamed
cards in such a way as to enforce the rules that had to be followed in w when the original black cards were constrained by
their respective suit stacks. Once the renamed images are released, all the auxiliary cards will be released to produce a win
in the target {do, O} instance.
This is done as follows. The FLAT-SOLITNoKing (1, 1) instance constructed will involve 3nb + 1 club cards and 3nb + 1 heart
cards. First, for O ≤ s < b, we rename the suit-(s + 1) cards in the instance w as follows:
C1,5 → 3ns + 3n0
C2,5 → 3ns+ 3n - 30
Cn-1,s → 3ns + 60
Cn,s → 3ns + 30.
```

### PDF page 7

```text
5258
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
Then, for O ≤ s < b, we add the n following build stacks, with the top card facing up and the bottom card (when present)
facing down:
Below:
Top:
3ns + 3n + 10
3ns + 3n - 200
3ns + 3n - 1d
3ns + 7%
..
3ns + 48
3ns + 8%
3ns + 58
Finally, the critical stack is set to the cards 3ns + 20b, 0 ≤ s < b, in any order, followed by the remaining cards
Ael, AS, 252, 30l, 450, 50, 6eb,..., 3nbob, 3nb + 18 in increasing order.
For anys, O ≤ s < b, until the AS becomes visible, no backward move is possible, and the cards 3ns + 3nS0, 3ns + 3n- 30,
...3ns + 68 and 3ns + 38 can only be placed in that order on the n build stacks designed to accept them. This holds for
each s independently. Only after the 3ns + 30 cards for O ≤ s < b have found their ways to their mates 3ns + 4% can
the critical stack be freed of the b cards 3ns + 2% sitting on top of it. In such an event, all the original build stacks arising
from the renamed w cards are empty and only sorted build stacks remain, leading to a win. This happens iff the original w
instance was winning. O
Corollary 9. FLAT-SOLITNoKing (1, 1) and SolIT(1, 1) are NL-hard.
The simplest Klondike restrictions can be solved by constant-depth circuits, as the following shows:
Theorem 10. (a) For any constant b, FLAT-SOLITNoKing (b, O) is in AC®
(b) FLAT-SOLIT (1, 0) is in AC°
(c) SoLIT (1, 0) is in AC [3].
Proof. Part (a): By Proposition 6, we need to determine whether a cycle exists in the graph H(w) of a nontrivial
FLAT-SOLITNoKing (b, O) instance w. We first prove it for b = 2 and explain later how to generalize the proof.
When b = 2, we claim that H(w) has a cycle iff there exist an edge (ido, ke) and an edge (jo, lob) in H(w) such that
i < landj < k. This condition is AC°-testable.
We now prove the claim. Call a pair of edges (ico, kG) and (j, leb) in H (w) a crossing when i < landj < k. Clearly, a
crossing together with the horizontal edges in H (w) form a cycle. Conversely, let G be a cycle in H(w). The cycle must have
cards from both suits otherwise the instance is trivial. Consider the two parallel paths, one for do and one for @, running
from left to right and formed by the horizontal edges in H(w). Let id and jo be the rightmost (i.e., lowest ranked) & and o
cards that belong to G. The edge in G leaving from ioo must lead to a do, say koo, otherwise the instance is trivial or i was not
the rightmost. Similarly, the edge in G leaving from jo must lead to a oo, say leo. It is not possible for both i = l andj = kto
hold, since no configuration of the build stacks can simultaneously give rise to the edges (ie, ja) and (jo, id). So assume
with no loss of generality that i < l. Then j < k otherwise the instance is trivial. So we have i < l andj < k.
For the case b > 2, we claim that H (w) has a cycle iff there exists a sequence of d ≤ b edges (Ca1,51, Cb2,52), (Ca2,52)
b3, 53), (Саз, s3, Cb4, 54),..., (Cad,Sd, Cb,sp) such that di ≤ Di tor 0 ≤ 1 > a. Clearly, these edges together with the horizontal
edges in H (w) creating paths from Cbi,s; to Caj,s; form a cycle. Conversely, assuming a cycle G, consider the edges leaving tron
the rightmost card trom G tor each suit involved in G in the order they appear in G. These edges have the claimed property.
This proves the claim and concludes part (a).
Part (b): Here we only have one suit, but the generalized King can refill an empty build stack. If the generalized King
occurs at the bottom of a build stack, then we accept iff the instance is nontrivial. Otherwise, we accept iff two conditions
hold:
• every build stack is sorted except the King's build stack which is sorted ignoring the King card, and
• if there is a card c above the King in the King's build stack, then there is another build stack all of whose cards are ranked
lower than c.
Part (c): Here we further have to deal with the pile and the talon. If r is the rank of a pile card c, we will denote by [r]
the rank of the largest ranked card in the pile ranked less than r (if no such card exists then let [r] = r). Suppose first that
the generalized King occurs in the build stacks. Then we accept iff the build stacks pass the test described in part b), and
furthermore, for each card c in the pile, say ranked r, we have:
1. if the card c' ranked [r] is above c when the pile is facing down, then one of the following conditions holds (where a
large card is defined as a card ranked higher than r):
• the number of large cards occurring between c' and c in the pile is congruent to 2 modulo 3, or
• the number of large cards on top of c in the pile is congruent to 2 modulo 3, or
• there are no large cards below c in the pile.
2. if the card c' ranked [r] is below c when the pile is facing down, then one of the following conditions holds:
• the number of large cards on top of c in the pile is congruent to 2 modulo 3, or
• there are no large cards between c' and c on the pile.
```

### PDF page 8

```text
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
Flat Klondike, no King
Flat Klondike
Klondike
1 black
in AC
in AC
in AC [3]
b blacks
in AC
in NL
in NL
* blacks
NL-complete
NL-hard, in NP
NL-hard, in NP
1 black, 1 red
NL-hard, in NP
NL-hard, in NP
NL-hard, in NP
2 blacks, 1 red
NP-complete
NP-complete
NP-complete
* blacks, * red
NP-complete
NP-complete
NP-complete
Fig. 2. Our current knowledge of the complexity of Klondike . A "b" represents any fixed number and an "*" represents an input-dependent number.
5259
These tests can be performed in parallel for each card c in AC® [3]. A case analysis and an induction show that these conditions
are necessary and sufficient to make a win possible.
Now suppose that the generalized King occurs in the pile. Then we first check that the instance without the pile is
nontrivial. Now let c' be the lowest ranked card sitting at the bottom of a build stack. Note that the option to move the
generalized King arises from the moment that c' is displaced to its suit stack. But the King need not be moved immediately:
it may be necessary to postpone moving the King until more pile cards have been displaced. This is solved by checking that
Vc King(c) holds, where c ranges over all pile cards ranked higher than c' and King(c) stands for the set of modular conditions
defined for c above, but now conceptually ranking the King between |rank(c)| and rank(c) in the card ordering.
Finally we note an upper bound that applies to the all-black instances:
Proposition 11. For any constant b, SoLIT(b, O) is in NL.
Proof. Consider a Solit(b, O) instance w. We can build a graph of configurations for the game starting from w. Indeed, note
that such a configuration can be deduced deterministically from w and the following data:
• the ranks of the highest ranked cards on top of the b suit stacks,
• the number of cards currently in the talon, and
• the set of Kings that have been moved to an empty build stack so far.
Since the number of suits is fixed, the total number of possible such data values is polynomial. Thus the configuration graph
can be computed in deterministic logspace from w. Then it remains to check in NL whether some configuration in which all
the Kings sit on the suit stacks is reachable from the initial configuration. •
5. Conclusion
Fig. 2 summarizes what we have learned in this work about the complexity of Klondike. Some gaps are obvious. In
particular, the cases involving two suits beg for a more satisfactory characterization. The flat cases of a red suit and a black
suit are especially puzzling. We strongly suspect these cases to be in P, but could they possibly be hard for P? Are they in
NL? Some simple Klondike cases involve the graphs H(w) built around a grid with the horizontal lines representing the suit
stack constraints. Could some of these be related to the grid graph reachability problems studied in [1]?
Our Klondike definition does not allow creating new build stacks in the course of the game, but the initial number of build
stacks is not bounded. Does Klondike remain NP-hard if we insist on only seven build stacks initially, as in the usual 52-card
Klondike? In the flat version, the number of configurations is then bounded by a polynomial and transitions between these
configurations are easy to compute, probably resulting in an NL upper bound. It would seem plausible that the general case
with seven stacks be doable in P as well.
Returning to our original motivations, we note on the one hand that Klondike and its restrictions can serve to illustrate
NP-completeness, but also a wealth of other complexity classes all the way down to AC®. On the other hand, Klondike
being NP-complete provides absolutely no mathematical justification that investigating the odds of winning in the case
of a standard 52-card deck will be difficult. But the fact that Klondike is just another name for SAT can at least be seen as
confirmation that the game does involve a good level of intricacy. Fig. 2 might suggest the following: start investigating the
odds of winning in the apparently simpler game restrictions and then proceed onwards to the NP-complete cases.
Acknowledgements
We thank François Laviolette from Laval University and Philippe Beaudoin from the University of British Columbia for
their help in proving the Klondike NP upper bound in Theorem 2. The second author thanks Andreas Krebs and Christoph
Behle in Tübingen for helpful discussions.
The second author was supported by the Natural Sciences and Engineering Research Council of Canada and the Fonds de
recherche sur la nature et les technologies du Québec.
References
[1] E. Allender, D. Barrington, T. Chakraborty, S. Datta, S. Roy, Grid graph reachability problems, in: Proc. 21st Annual IEEE Conference on Computational
Complexity, pp. 299-313, 2006.
[2] D. Barrington, N. Immerman, H. Straubing, On uniformity within NC', Journal of Computer and System Sciences 41 (3) (1990) 274-306.
```

### PDF page 9

```text
5260
L. Longpré, P. McKenzie / Theoretical Computer Science 410 (2009) 5252-5260
[3] S. Buss, S. Cook, A. Gupta, V. Ramachandran, An optimal parallel algorithm for formula evaluation, SIAM Journal on Computing 21 (1992) 755-780.
14] Malte Helmert, Complexity results for standard benchmark domains in planning, Artificial Intelligence 143 (2) (2003) 219-262.
[5] I. Gent, C. Jefferson, I. Lynce, I. Miguel, P. Nightingale, B. Smith, A. Tarim, Search in the Patience Game "Black Hole", in: Al Communications, ISSN:
0921-7126, IOS Press, pp. 1-15.
[6] R. Kaye, Minesweeper is NP-complete, in: The Mathematical Intelligencer, vol. 22, no. 2, Springer Verlag, 2000, pp. 9-15.
[7] C. Papadimitriou, Computational Complexity, Addison-Wesley, 1994.
[8] D. Parlett, Solitaire: Aces Up and 399 Other Card Games, Pantheon, 1979.
[9] D. Parlett, A History of Card Games, Oxford University Press, 1991.
[10] X. Yan, P. Diaconis, P. Rusmevichientong, B. Van Roy, Solitaire: Man Versus Machine, in: Proc. Advances in Neural Information Processing Systems, 17,
```
