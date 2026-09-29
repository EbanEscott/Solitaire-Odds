# Counting Classes of Klondike Solitaire Configurations

- **Citation key:** `deruiter2012counting`
- **Authors:** Johan de Ruiter
- **Publication:** Leiden University master’s thesis, Internal Report 2012–9; title page dated 23 August 2012; 63 PDF pages. Printed page numbers are one less than the physical PDF page numbers.
- **Local PDF:** [Counting Classes of Klondike Solitaire Configurations](counting-classes-of-klondike-solitaire-configurations.pdf)
- **Source and version:** [Leiden repository](https://theses.liacs.nl/pdf/2012-09JohandeRuiter.pdf); MSc thesis, August 2012.
- **Downloaded:** 2026-09-29
- **PDF pages:** 63

## Summary

Studies structural classes of Klondike deals through exact counting, dynamic programming, inclusion–exclusion, and simulation. It also investigates stock-access restrictions and classes of trivially solvable configurations.

## Key findings

- Counts a class of within-column blocking configurations comprising approximately 1.181069% of all deals (Section 3.5, printed p. 40 / PDF p. 41).
- Finds 217,328 deals with broader type-II blocking patterns among 10,000,000 sampled deals, approximately 2.17% (Section 3.6, printed p. 41 / PDF p. 42).
- Unplayable and selected single-move classes together establish another lower bound of about 0.3645% unwinnable deals (conclusion, printed p. 59 / PDF p. 60).

## Conditions and limitations

These are particular classes of unwinnable deals, not a complete estimate of winnability or player performance. The classes may overlap, so their percentages must not be added. Type-II frequency is simulated rather than counted exactly. The no-braider analysis grants access to every stock card (printed p. 60 / PDF p. 61).

## Relevance to Solitaire Odds

Our interpretation: useful for constructing structural deadlock diagnostics and distinguishing provably impossible deals from failures caused by player decisions.

## Extracted PDF text

Apple PDFKit text extraction, extracted 2026-09-29. Page headings below use physical PDF page numbers, including covers and front matter; printed page numbers can differ. Automatic extraction can lose table layout, equations, card symbols, and figure labels; consult the PDF for exact notation. Unmapped control characters are shown as `�`. The text below is source material, separate from our summary and interpretation above.

### PDF page 1

```text
Internal Report 2012–9 August 2012
Universiteit Leiden
Opleiding Informatica
Counting Classes
of
Klondike Solitaire Configurations
Johan de Ruiter
MASTER’S THESIS
Leiden Institute of Advanced Computer Science (LIACS)
Leiden University
Niels Bohrweg 1
2333 CA Leiden
The Netherlands
```

### PDF page 2

```text
Counting Classes
of
Klondike Solitaire Configurations
Johan de Ruiter, johan.de.ruiter@gmail.com
August 23, 2012
Abstract
Klondike Solitaire – also known as Patience – is a well-known sin-
gle player card game. We studied several classes of Klondike Solitaire
game configurations. We present a dynamic programming solution
for counting the number of “unplayable” games. This method is ex-
tended for a subset of games which allow exactly one move. With
an algorithm based on the inclusion-exclusion principle, symmetry
elimination and a trade-off between lookup tables and dynamic pro-
gramming we count the number of games that cannot be won due
to a specific type of conflict. The size of a larger class of conflicting
configurations is approximated with a Monte Carlo simulation. We
investigate how much gameplay is limited by the stock. We give a re-
cursion and show that Pfaff-Fuss-Catalan is a lower bound. We con-
sider trivial games and report on two remarkable patterns we discov-
ered.
1
```

### PDF page 3

```text
Preface
The research presented in this thesis was done in pursuit of a Master of
Science degree in Computer Science from Leiden University in the Nether-
lands. The topic Klondike Solitaire was suggested to me by my supervisor,
dr. Walter Kosters.
Originally I planned to focus my efforts on empirically determining
bounds on the fraction of solvable Klondike Solitaire games for several
different rule sets, but I was ultimately much more interested in the enu-
merative combinatorial aspects of the game. And so it came to pass, as
other questions came to mind, and branched, and branched, that the end
result turned out quite different.
I have been into algorithms ever since I learned of the Dutch Olympiad
in Informatics, back in high school. For my 17th birthday my parents gave
me a copy of Steven S. Skiena’s The Algorithm Design Manual [1]. Especially
Skiena’s War Stories were a delight to read. I was fascinated by how seem-
ingly too computationally intensive problems could be broken down with
clever algorithms, datastructures and optimizations. Now, it would almost
be sacreligious to compare my own work with that of Skiena, but I hope
to have at least at times captured the spirit of the War Stories. This should
become most apparent in the chapter Counting Blocked Klondike Solitaire
Deals, where we pile a variety of techniques on top of one another in order
to compute a 66 digit number within a reasonable amount of time.
A second influential factor, largely by proxy, has been Richard Bell-
man’s Dynamic Programming [2]. In training for the Benelux Algorithm
Programming Contest (3rd place 2006, winner 2007 and 2008, 4th place
2009) and the North-western European Regional Contest of the ACM ICPC
(3rd place 2008), I completed near to 1,500 programming exercises. These
tasks have deeply familiarized me with dynamic programming and its
wide range of applications. In the chapter Counting Unplayable Klondike
Solitaire Deals we present a very fast dynamic programming solution where
previously only a much slower algorithm had been applied. The technique
also appears, in more of a supporting role, elsewhere in the text.
During the later stages of my research I became increasingly interested
in Generating Functions (a great resource is Wilf’s Generatingfunctionol-
ogy [3]) and their possible applications to the chapter Counting No-braider
Klondike Solitaire Deals. Unfortunately, I have not yet been able to success-
fully apply them.
2
```

### PDF page 4

```text
Acknowledgements
This work was done under the supervision of dr. Walter Kosters of LIACS,
Leiden University. I want to thank Walter for our effective brainstorm ses-
sions and for all the fun we had. Let me also express my gratitude to my
second supervisor, dr. Hendrik Jan Hoogeboom and to dr. Jeannette de
Graaf, both of LIACS, Leiden University. I thank Maarten L¨ offler for proof-
reading.
Next, I would like to thank my parents, Hans and Heleen, for their
love, for their support, for their patience. I want to thank Shani for being
the best possible influence. Thank you, Misha and Thomas, for sharing my
passion for algorithm contests and problem solving, and for your great
efforts to make us excel, both as individuals and as a team. Thank you,
Martin, Harm, Arjan, Mirjan, Maarten, Daniel, for always sticking around.
Thank you, everyone with whom I had the pleasure of exchanging
ideas throughout the years. Our diversions have been most valuable to
me.
3
```

### PDF page 5

```text
Contents
1 Introduction to Klondike Solitaire 6
1.1 Rules and Terminology . . . . . . . . . . . . . . . . . . . . . . 6
2 Counting Unplayable Klondike Solitaire Deals 8
2.1 2.2 Conditions and Campaign Plan . . . . . . . . . . . . . . . . . 8
A Dynamic Programming Solution . . . . . . . . . . . . . . . 9
2.2.1 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.3 Single Move Klondike Solitaire Games . . . . . . . . . . . . . 11
2.3.1 Category A . . . . . . . . . . . . . . . . . . . . . . . . 11
2.3.2 Category B . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.3.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
3 Counting Blocked Klondike Solitaire Deals 15
3.1 Type I Blocked Klondike Solitaire Deals . . . . . . . . . . . . 15
3.2 Inclusion-Exclusion . . . . . . . . . . . . . . . . . . . . . . . . 17
3.3 Composite Restricted Blocking Patterns . . . . . . . . . . . . 19
3.3.1 A single primitive blocking pattern . . . . . . . . . . 21
3.3.2 Compositions of two primitive blocking patterns . . 22
3.3.3 Compositions of three primitive blocking patterns . . 26
3.3.4 Compositions of four primitive blocking patterns . . 27
3.4 Composite Restricted Blocking Pattern Aggregation . . . . . 28
3.4.1 Quadruples of tuples . . . . . . . . . . . . . . . . . . . 29
3.4.2 Factorial base counter . . . . . . . . . . . . . . . . . . 29
3.4.3 Bitwise representation and nested loops . . . . . . . . 30
3.4.4 Exploiting symmetry . . . . . . . . . . . . . . . . . . . 30
3.4.5 Lookup tables and dynamic programming . . . . . . 31
3.4.6 A shattered lookup table . . . . . . . . . . . . . . . . . 34
3.4.7 Intermezzo: Generalizing shattered lookup tables . . 36
3.4.8 Final notes on exploiting symmetry . . . . . . . . . . 38
3.5 3.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
Extension to Type II . . . . . . . . . . . . . . . . . . . . . . . . 40
4 Klondike Draws 42
4.1 4.2 4.3 4.4 Deriving a Recurrence . . . . . . . . . . . . . . . . . . . . . . 42
Klondike Draws with k<n≤2k. . . . . . . . . . . . . . . . 44
Klondike Draws with 2k<n≤3k. . . . . . . . . . . . . . . 45
Pfaff-Fuss-Catalan as a Lower Bound . . . . . . . . . . . . . . 45
4
```

### PDF page 6

```text
5 Counting No-braider Klondike Solitaire Deals 48
5.1 Braid Numbers . . . . . . . . . . . . . . . . . . . . . . . . . . 48
5.2 A Recursive Definition for B((h1,h2) →(ℓ1,ℓ2)). . . . . . . . 50
5.3 A Case Study: B((h,h) →(h,h)). . . . . . . . . . . . . . . . . 50
5.4 An Algorithm to Compute Braid Numbers . . . . . . . . . . 52
5.5 A Graph Theoretical Interpretation . . . . . . . . . . . . . . . 53
5.6 A Case Study: B(R2,k →R2,k). . . . . . . . . . . . . . . . . . 54
5.6.1 A recursive definition and a closed formula . . . . . . 54
5.7 Application of Braid Numbers to Klondike Solitaire . . . . . 57
6 Conclusions and Future Research 59
Bibliography 61
5
```

### PDF page 7

```text
1 Introduction to Klondike Solitaire
Klondike Solitaire, sometimes referred to as Patience or simply Solitaire is
such a well-known game that it hardly requires an introduction. For one
thing, the game has been shipped with every version of Windows since
Windows 3.0 [4], and of Windows 7 alone, over 525 million licenses have
been sold [5].
Interestingly enough though, it has been said that “it is one of the em-
barrassments of applied mathematics that we cannot determine the odds
of winning the common game of solitaire” [6]. It has been empirically de-
termined that at least 82% and at most 91.44% of Klondike Solitaire games
have winning solutions [7].
1.1 Rules and Terminology
When we speak of the standard rules of Klondike Solitaire, we are referring
to the default rules in Solitaire for Windows [8]. The terminology used dif-
fers slightly between different articles in the literature, so we will specify
the definitions used throughout this text.
For reference, Figure 1 shows a starting configuration of a game of Soli-
taire in Windows 7.
Figure 1: A screenshot of Solitaire in Windows 7 [8].
6
```

### PDF page 8

```text
Klondike Solitaire is played with a regular deck of 52 playing cards.
Such a deck contains cards of four suits: the red suits hearts (♥) and dia-
monds (♦), and the black suits clubs (♣) and spades (♠). There are thirteen
cards of each suit, commonly named (or ranked) ace (A), 2, 3, 4, 5, 6, 7, 8, 9,
10, jack (J), queen (Q) and king (K). In the context of Solitaire they might as
well be regarded as numbers from 1 through 13.
The foundation – the area in the top right – is where cards can be stacked
upon cards of the same suit, strictly increasing by one each time and start-
ing with the ace. An ace can be put only at an empty location. The goal in
Klondike Solitaire is to move all cards to the foundation.
The lower region of the screen is called the tableau. At the start of the
game there are seven piles of cards, which have heights 1 through 7. For
each pile the top card (i.e., the card dealt last) is dealt face up, while the
other cards are dealt face down. Whenever the topmost face down card of
a pile is exposed, it is turned over. Cards that face down are never moved
around. A card may be moved on top of a face up card on the tableau,
provided that the colors or their suits are different and that the rank of
the moving card is one lower than the rank of the other card. One can
also move groups around on the tableau, provided that the bottom most
card (i.e., the one that will be placed directly on top of another pile) is of a
different suit color and has rank one lower than the card it is moved upon.
A king might be moved to an empty pile, as long as afterwards there are
at most seven piles.
At this point different rule sets start to diverge. Bjarnason, Tadepalli
and Fern [7] hold true to the rules of Solitaire in Windows 7, by allowing
the player to move any number of topmost face up cards from a single pile
at once, while Yan, Diaconis, Rusmevichientong and Van Roy [6] expressly
state that when moving multiple cards at once, all face up cards from the
source pile must be moved together. Within this text this distinction is ir-
relevant.
In the upper left part of the screen we find the stock. At the start of
the game it consists of 24 cards facing down. The player can flip the cards
of the stock over, three at a time, onto a new pile; the waste. The topmost
card of the waste can be played to the foundation or to the tableau if the
other rules do not prevent this. Once the stock is empty, the waste may
be turned upside down at the stock location and reused. Most commonly
this action may be performed an unlimited number of times (Bjarnason et
al. [7]), although some rule sets limit the player to cycle through the stock
only once or three times.
Solitaire for Windows 7 has an option to flip cards per one rather than
per three. Where relevant we will indicate which variation we consider.
7
```

### PDF page 9

```text
2 Counting Unplayable Klondike Solitaire Deals
It was reported by Latif [9] that unplayable games in Klondike Solitaire,
i.e., games in which not a single move can be made, amount to approxi-
mately 0.25% of all possible games. Donkersteeg and Kosters reported [10]
that they managed to compute the exact ratio of unplayable Klondike Soli-
taire games in about one minute on a 3 GHz processor with a brute force
approach and using smart optimizations (≈200 lines of C++ code). We
confirm the value reported in [10] with a dynamic programming solution
that runs in mere milliseconds and consists of only a couple of lines of C++
code.
Our algorithm scales linearly in both time and space for larger instances
of the problem (more cards per suit, more piles on the table and more avail-
able cards in the stock), while it is expected that the approach followed by
Donkersteeg and Kosters will quickly cease to be feasible as the problem
size increases.
2.1 Conditions and Campaign Plan
As laid out in [10], a necessary and sufficient set of conditions for a game
to be unplayable is:
•There are no aces among the accessible stock cards.
•There are no aces among the accessible tableau cards.
•There are no two accessible cards on the tableau of suits with a dif-
ferent color and with ranks differing by one.
•There is no accessible card in the stock which has a rank that is one
less than an accessible tableau card of a suit of the opposite color.
In the standard version of Klondike Solitaire the number of accessible
cards on the tableau at the time of dealing is seven and the number of
accessible cards in the stock is eight. The number of ranks per suit in a reg-
ular deck of playing cards is thirteen. The number of accessible cards in
the stock follows from the fact that there are 24 cards in the stock, but only
every third card can be accessed. Under the circumstances it is irrelevant
whether one is allowed to go through the stock a limited or an unlimited
number of times.
When there are piles cards accessible on the tableau and stock cards ac-
cessible in the stock, the ratio of unplayable games equals the number of
8
```

### PDF page 10

```text
ways one can choose piles cards to be accessible on the tableau and stock
cards to be accessible in the stock, while satisfying the above conditions,
divided by the total number of ways one can choose piles cards to be ac-
cessible on the table and stock cards to be accessible on the stock. It is
the numerator N(ranks,piles,stock ) that we will compute with a dynamic
programming approach. With a deck size of 4 ×ranks, the denumerator
equals:
(4 ×ranks)!
piles! ×stock ! ×(4 ×ranks−piles−stock )!
2.2 A Dynamic Programming Solution
We define M(ranks,piles,stock ,r,b) to be number of ways one can choose
piles cards to be accessible on the tableau and stock cards to be accessible
in the stock, using a deck with size 4×ranks, with rred cards of rank ranks
and bblack cards of rank ranks, while satisfying the conditions stated in
the previous section. From the fact that there are two cards for each com-
bination of rank and suit color, it follows that:
N(ranks,piles,stock ) =
2
2
M(ranks,piles,stock ,r,b)
r=0
b=0
The function M allows for the following recursive definition:
M(n,p,s,r,b) =
r
b
2
2
rp=0
bp=0
r′=0
b′=0
2f cM(n−1,p−pr−pb,s−sr−sb,r′,b′)
for n,p,s ≥2 and 0 ≤r,b ≤2, and 0 otherwise, except M(0,0,0,0,0) = 1
and M(1,0,0,0,0) = 1. We define sr = r−pr, sb = b−pb and f= f1 + f2,
with:
f1 =
f2 =
1 if pr = 1 ∨sr = 1
0 otherwise
1 if pb = 1 ∨sb = 1
0 otherwise
Also:
c=
1 if (r′= 0 ∨pb = 0) ∧(b′= 0 ∨pr = 0)
0 otherwise
We will discuss the rationale behind the recursive definition of M. Any
possible choice of pand scards from those cards with rank at most n(ex-
cept the aces, and satisfying the previously specified conditions), contain-
ing rred cards of rank nand bblack cards of rank n, can be built from valid
9
```

### PDF page 11

```text
choices of cards with rank at most n−1. We should distinguish between
red cards of rank n that have been selected for the stock and those that
have been selected for the table, say r= pr + sr. Likewise, b= pb + sb. Log-
ically, this leaves p−pb−pr cards to be picked for the piles and s−sb−sr
cards to be picked for in the stock. If we pick no red cards of rank n, or
if we select two red cards of rank n to be in the same category (piles or
stock), we do not have to choose between them, but if in one or both of the
categories we pick one red card of rank n, we do need to make a choice,
and this would involve a multiplication with a factor 2. This is reflected in
the formula by 2f , which has been broken down into f1 and f2, pertaining
to the red and the black cards of rank nrespectively. An important condi-
tion for the validity is that we do not choose any black cards of rank nif
we have selected red cards of rank n−1 and no red cards of rank nif we
have selected black cards of rank n−1. This is why we iterate over r′and
b′from 0 to 2 and this is where the conditional factor ccomes into play. It
is 1 if and only if there is no such conflict, and 0 otherwise.
To avoid choosing aces, we set M(1,0,0,0,0) to 1 and start iterating
over nat n= 2.
Both the space and time complexity of this algorithm are O(rps), mean-
ing it scales only linearly in each of the dimensions of the problem. As
a consequence, we can compute N(13,7,8) exactly in a matter of milisec-
onds, which is a tremendous improvement over the algorithms previously
in existence.
Figure 2 shows a C++ implementation of the proposed algorithm.
Note that in fact we do not have to iterate over the the number of red
cards (respectively black cards) of rank n−1. Instead, we could iterate
over none and some. We would still iterate over the number of red cards
(respectively black cards) of rank n, but we would aggregate the results
for one and two cards as some. This would reduce the amount of memory
required by a factor 9
4 and the amount of time spent by a factor 3
2.
2.2.1 Results
The value we have computed for N(13,7,8) is 72,099,595,172,416, which
confirms the result found by Donkersteeg and Kosters [10] – approximately
0.2500186% of all games is unplayable (about one in 400).
The version of Solitaire in Windows Vista and Windows 7 has an option
to flip the cards in the stock one at a time, meaning all cards in the stock
are accessible. Using N(13,7,24) = 89,367,495,137,280, we determine that
in this case the fraction of unplayable games is 89, 367, 495, 137, 280
52!/(7!×24!×21!) , or approxi-
mately 0.0000177% (about one in 5,649,223).
10
```

### PDF page 12

```text
1 long long N( i n t ranks , i n t p i l e s , i n t stock ) {
2 long long X=0 , M[ ranks + 1] [ p i l e s + 1 ][ stock + 1 ] [ 3 ] [ 3 ] ;
3 memset (M, 0 , si z e o f (M) ) ;
4 M[ 0 ] [ 0 ] [ 0 ] [ 0 ] [ 0 ] =M[ 1 ] [ 0 ] [ 0 ] [ 0 ] [ 0 ] = 1 ;
5
10 11 12 13 14 15 16 17 18 19
20 21 22 23 }
6 for ( i n t n=2;n<=ranks ; n++)
7 for ( i n t p=0;p<=p i l e s ; p++)
8 for ( i n t s =0; s<=stock ; s ++)
9 for ( i n t r =0; r <=2; r ++)
for ( i n t b =0; b<=2;b++)
for ( i n t s r =0 , pr=r ; sr <=r ; s r ++ ,pr−−)
for ( i n t sb =0 ,pb=b ; sb<=b ; sb ++ ,pb−−)
i f ( s r+sb<=s&&pr+pb<=p )
for ( i n t r =0; r <=2; r ++) i f ( ! ( r&&pb ) )
for ( i n t b =0; b <=2; b ++) i f ( ! ( b&&pr ) )
M[ n ] [ p ] [ s ] [ r ] [ b]+=
M[ n−1][p−pr−pb ] [ s−sr−sb ] [ r ] [ b]<<
( ( pr ==1||s r ==1) +(pb ==1||sb ==1) ) ;
for ( i n t i =0; i <9; i ++)
X+=M[ ranks ] [ p i l e s ] [ stock ] [ i / 3] [ i %3];
return X ;
Figure 2: An implementation of a dynamic programming algorithm to
compute N(ranks,piles,stock ).
2.3 Single Move Klondike Solitaire Games
By extension, we can distinguish several classes of Klondike Solitaire games
in which no more move is possible after any opening move – assuming
moves cannot or will not be undone. The classes we will examine more
closely are those in which exactly one ace is available – on the tableau or
in the stock (category A and category B respectively) – which can be moved
to the foundation, after which no other move can be performed, not even
when taking into account the possibly newly available card on the tableau
in case the ace was moved from a pile of height 2 or higher, and the poten-
tially newly available cards in the stock in case the ace originated there.
2.3.1 Category A
Let us first explore the scenario in which there is exactly one ace among
the seven cards accessible on the table and none elsewhere. There are six
possible positions for which playing this ace will expose an extra card.
11
```

### PDF page 13

```text
Only when the ace is located in the pile of height 1, this will not be the case.
Furthermore, if the ace is located in the pile of height 1 there should not
be a king among the accessible cards, since this would allow for moving a
king to the empty pile after the ace has been moved to the foundation.
The ace can be of any of the four suits and the situation is completely
symmetrical in this respect. Whichever ace we are dealing with, neither of
the two 2’s of the same color should be among the remaining cards.
If we define N(n,p,s,P,S,Q) to be the number of ways one can select
p cards to be accessible on the table and s cards to be accessible in the
stock from a deck with ranks from 1 through n, while choosing the cards
from the set P among those on the table, choosing the cards from the set S
among those in the stock, and not choosing any of the cards from the set
Q, the number of configurations in category A equals:
CA = 4 ×6 ×N(13,8,8,{♥A},∅,{♦A,♣A,♠A}) ×7! ×8! ×36!
+ 4 ×N(12,7,8,{♥A},∅,{♦A,♣A,♠A}) ×6! ×8! ×37!
In a similar vein, we express the number of unplayable games as:
C0 = N(13,7,8,∅,∅,{♥A,♦A,♣A,♠A}) ×7! ×8! ×37!
The factorials have been carefully chosen to accomodate the permuta-
tions of the accessible cards in the stock, the accessible cards on the tableau
(except for the ace) and all inaccessible cards. The factor 4 stems from the
symmetry among the aces.
2.3.2 Category B
In the scenario of category B there is exactly one ace present among the
eight cards accessible in the stock and none elsewhere. The number of
cards accessible in the stock after playing this ace to the foundation de-
pends on the position of the ace in the stock.
Figure 3 shows a schematic representation of the accessible cards in the
stock with one ace present. Initially every third card can be accessed. Be-
sides the ace these are the cards displayed in blue. Directly after the ace,
which is the twelfth card from the top (1-based), has been moved to the
foundation, the eleventh card becomes accessible. All blue cards remain
available after the ace has been moved. Now if we decide to cycle through
the entire stock and start turning groups of three cards over from the be-
ginning, things have changed. Every card that was at a position p ≡ 1
12
```

### PDF page 14

```text
Figure 3: A schematic representation of the accessible cards in the stock
when playing an ace from the stock to the foundation. The topmost card
in the stock is displayed in the upper left. The cards with a blue back rep-
resent the cards accessible before the ace has been played, while the cards
with a red back represent the cards newly accessible after the ace has been
moved to the foundation. All playing card images used in the figures in
this thesis were created by Aguilar [11].
(mod 3) with p > 12 before, has been shifted into a position that is a mul-
tiple of three. These are the red cards positioned to the right of the ace in
Figure 3.
More quantitatively speaking, the ace can be located at any position
3 ×ifor i ∈[1,8], and for such a position 3 ×ithere will be 16−icards
accessible in the stock after the ace has been played to the foundation. The
number of games involved in this scenario can therefore be expressed as:
CB = 4×
8
i=1
N(13,7,16−i,∅,{♥A},{♦A,♣A,♠A})×7!×(16−i)!×(28+i)!
2.3.3 Results
To compute N for any sets P, S and Q, the program from Figure 2 would
need to be modified to take into account cards not to process and mul-
tiplication factors f1, f2 not to be applied. However, in our case we are
solely working with special cases involving aces. Therefore, it will suffice
to modify the values to be initialized for n= 1 and to start iterating over n
at n= 2. For category A we initialize M(1,1,0,1,0) as 1, while for category
B we initialize M(1,0,1,1,0) as 1.
Table 1 lists the queries of the function N that are used to compute the
13
```

### PDF page 15

```text
class query result
C0 N(13,7,8,∅,∅,{♥A,♦A,♣A,♠A}) 72,099,595,172,416
CA N(12,7,8,{♥A},∅,{♦A,♣A,♠A}) 3,845,001,461,416
N(13,8,8,{♥A},∅,{♦A,♣A,♠A}) 40,275,958,345,024
CB N(13,7,8,∅,{♥A},{♦A,♣A,♠A}) 14,069,684,839,712
N(13,7,9,∅,{♥A},{♦A,♣A,♠A}) 40,275,958,345,024
N(13,7,10,∅,{♥A},{♦A,♣A,♠A}) 98,528,491,384,320
N(13,7,11,∅,{♥A},{♦A,♣A,♠A}) 208,273,411,645,728
N(13,7,12,∅,{♥A},{♦A,♣A,♠A}) 383,699,702,412,000
N(13,7,13,∅,{♥A},{♦A,♣A,♠A}) 620,236,030,645,920
N(13,7,14,∅,{♥A},{♦A,♣A,♠A}) 884,347,045,037,280
N(13,7,15,∅,{♥A},{♦A,♣A,♠A}) 1,116,793,930,397,280
Table 1: The queries of the function N that are used to compute the number
of unplayable games and the number of games of category A and category
B.
number of unplayable games and the number of games of category A and
category B. When we insert the values listed in the table into the formulas
derived earlier, we find that CA
52! = 0.00098212...and CB
52! = 0.00016328....
Since C0 + CA + CB constitutes a lower bound on the number of games
that allow at most one move, we can conclude that at least 0.3645...% of
all games allows at most one move.
14
```

### PDF page 16

```text
3 Counting Blocked Klondike Solitaire Deals
Some compositions of the tableau in Klondike Solitaire prohibit successful
completion of a game even when playing with very relaxed sets of rules
and with complete information about the locations of all the cards. We
count the number of games which can not be won due to the presence of
a specific type of conflict.
3.1 Type I Blocked Klondike Solitaire Deals
Consider a situation immediately after dealing where a 7 of hearts (♥7)
blocks access to the 8 of spades (♠8), the 8 of clubs (♣8) and the 3 of hearts
(♥3) by having been dealt on top of them. The ♥7 cannot be moved to the
foundation before the ♥3 has been moved to the foundation, but the ♥3
cannot be moved to the foundation before the ♥7 has been moved else-
where. The ♥7 can not be moved to an empty position, because it is not
a king and it cannot be moved on top of another pile because both of the
black 8’s are inaccessible as long as the ♥7 has not been moved.
In short, the ♥7 has nowhere to go and the game cannot be won — it
is blocked. In this chapter we will describe a reasonably fast algorithm to
compute the exact fraction of games that cannot be won precisely because
this type of conflict is present immediately after the cards have been dealt.
More generally spoken, when within a single pile we have a card with
value xof a suit sthat blocks access to the two cards with value x+ 1 of
the color that is not the color of the suit sand to a card with value yof the
suit swith y < x, the game is blocked. Any such (unordered) set of four
cards we will call a primitive blocking set. The card with value xis said to
act as the lock. The primitive blocking set mentioned in the example above
is illustrated in Figure 4. Here the ♥7 is the lock.
Figure 4: A primitive blocking set with ♥7 as its lock. As the word set
suggests, the order of the cards does not matter. However, note that ♥7
should be on top of the other cards in the primitive blocking set for the set
to block the game.
A regular deck of playing cards has thirteen cards of each of four suits
15
```

### PDF page 17

```text
and therefore contains
4
12
(x−1) = 264
x=2
primitive blocking sets.
In a pile of height hthere are h
4 subsets of positions that could poten-
tially be occupied by a primitive blocking set. For a given subset of four
positions in a pile, there are 3! ways the cards of a primitive blocking set
can be assigned to these positions such that the game is blocked, because
the lock card must be dealt after the other three cards, but otherwise the
order is free. It follows that there are
3! h
4
×4
12
x=2
(x−1) = 1,584 h
4
ways to assign a primitive blocking set to positions in a pile of height h.
Figure 5: A primitive blocking pattern, as depicted here, is the assignment
of the cards of a primitive blocking set to locations in the tableau, such that
we can tell the deck is blocking without knowing any of the other cards.
All cards in this figure and the figures to follow within this chapter are
facing down, even when displayed facing up.
We call the assignment of the cards in a primitive blocking set to posi-
tions in a certain pile on the tableau a primitive blocking pattern. An exam-
ple is shown in Figure 5. As primitive blocking patterns need to have their
cards assigned within a single pile with height at least 4, the total number
of distinct primitive blocking patterns is
1,584
7
h=4
h
4 = 88,704.
16
```

### PDF page 18

```text
3.2 Inclusion-Exclusion
We are interested in counting the number of deals in which at least one
primitive blocking pattern is present. To count this number we use an ap-
proach based on the inclusion-exclusion principle. The inclusion-exclusion
principle states that the number of elements in the union of two finite sets
equals the sum of the number of elements for each of the sets minus the
number of elements in their intersection, that is, |A∪B|= |A|+|B|−|A∩B|
for sets Aand B.
This statement can be generalized to
n
i=1
Ai =
n
|Ai|−
|Ai ∩Aj |
i=1
i,j:1≤i<j≤n
+
|Ai ∩Aj ∩Ak|−...+ (−1)n−1|A1 ∩...∩An|
i,j,k:1≤i<j<k≤n
where A1,A2,...,An are sets [12].
The sets we concern ourselves with in this counting problem are 88,704
subsets of the set of all 52! possible deals. Each of these 88,704 subsets is
associated with a single primitive blocking pattern: it contains these deals
that have the four cards of the associated primitive blocking pattern as-
signed to the designated locations. We will call such a subset a simple re-
stricted set. The term restricted derives from the fact that all the assigned
cards are assigned to locations within the same pile. Primitive blocking
patterns or simple restricted sets which have their cards assigned to the
same pile, we will call co-restricted. Each of the simple restricted sets Ai
with 1 ≤i ≤88,704 has (52−4)! = 48! elements, because four cards are
fixed and all others can be freely permuted.
To bluntly evaluate all 288, 704 intersections of subsets of the 88,704 sim-
ple restricted sets, however, would clearly be infeasible.
We observe that for many tuples of simple restricted sets the intersec-
tion has to be empty. This is always due to one or both of the following
reasons:
1. Two or more of the sets in the tuple are associated with primitive
blocking patterns which have assigned the same card to a different
location.
2. Two or more of the sets in the tuple are associated with primitive
blocking patterns which have assigned different cards to the same
location.
17
```

### PDF page 19

```text
We can write every intersection clause in the inclusion-exclusion for-
mula as the intersection of four possibly empty sets, each being the inter-
section of simple restricted sets in the clause that are co-restricted, because
there are only four interesting piles. Such a composition by means of in-
tersection of one or more simple restricted sets associated with the same
pile, we call a composite restricted set.
At the same time, every quadruple of composite restricted sets (here we
consider two composite restricted sets distinct if they were formed from
a different set of simple restricted sets, even if they contain exactly the
same elements) associated with different piles corresponds with exactly
one clause in the inclusion-exclusion formula. The sign of a clause corre-
sponding to a quadruple of composite restricted sets can be determined
from annotations, provided that we annotate each composite restricted set
with an indication of whether it was constructed from an odd or an even
number of simple restricted sets.
Any blocking pattern arising from combining primitive blocking pat-
terns that have no conflicts (i.e., they do not assign the same card to a dif-
ferent location on the tableau and they do not assign different cards to the
same location on the tableau), regardless if they have their cards located
within the same pile or not, we call a composite blocking pattern. In par-
ticular, composite restricted sets are associated with composite blocking
patterns that combine the card-to-location assignments of the individual
primitive blocking patterns.
Figure 6: Four piles of height 7 which were extracted from tableaus in
which no other cards were assigned to locations. Hence, the leftmost pile
can be said to represent a composite blocking pattern, whereas the other
piles represent primitive blocking patterns.
Because some composite blocking patterns can be the result of different
combinations of primitive blocking patterns (in particular, some compos-
18
```

### PDF page 20

```text
ite blocking patterns can be the result both of combining an odd and of
combining an even number of primitive blocking patterns), it is best to
maintain counts of occurrences of tuples containing a composite blocking
pattern and the parity of the number of primitive blocking patterns used
to construct it.
As an example, consider Figure 6. Each of the four piles in Figure 6
was extracted from a tableau in which no other cards were assigned to lo-
cations. The leftmost pile represents a composite blocking pattern that can
be obtained from combining the other piles, which all represent primitive
blocking patterns, in three ways: The second pile could be combined with
third pile, the second pile could be combined with the fourth pile, or the
second pile could be combined with the third and the fourth pile.
3.3 Composite Restricted Blocking Patterns
In line with other definitions, with the term composite restricted blocking
pattern we refer to a composite blocking pattern composed of co-restricted
primitive blocking patterns.
We observe that a composite restricted blocking pattern restricted to a
pile of height 7 can contain at most four distinct primitive blocking pat-
terns. A pile of height 6 can contain at most three primitive blocking pat-
terns, while in a pile of height 5 there can be only two and in a pile of
height 4 there can be only one. This suggests a recursive approach with a
recursion depth of at most ten, with an upper bound of evaluating
7
h=4
h−3
k=0
1,584 h
4
k
≈5.6 ×1084
compositions of primitive blocking patterns. Although this is an enor-
mous improvement over having to evaluate 288, 704, it would still be com-
pletely infeasible for any modern computer unless we could do a signifi-
cant amount of backtracking.
We investigate the number of composite restricted blocking patterns to
gain more insight into the feasibility. Table 2 shows in how many ways
different numbers of primitive blocking patterns can be combined with-
out conflicts within piles of height 4 through 7 as computed by an elab-
orate enumeration by hand and confirmed by an exhaustive search by a
computer.
By multiplying the entries from the row with totals in Table 2 we get
a much more reasonable value for the number of composite blocking pat-
terns to evaluate, provided that we precompute all valid composite re-
19
```

### PDF page 21

```text
#patterns \height 4 5 6 7
0 1 1 1 1
1 1,584 7,920 23,760 55,440
2 21,120 187,440 2,452,560
3 237,600 4,316,400
4 2,280,960
total 1,585 29,041 448,801 9,105,361
Table 2: The number of valid compositions of distinct primitive blocking
patterns classified according to the number of primitive blocking patterns
used and the height of the pile in which they are used. Empty cells denote
zeros.
stricted blocking patterns. The resulting number, 1.9 ×1020, is somewhat
reassuring, although it is still too large.
A very important observation at this point is that once we have com-
bined primitive blocking patterns restricted to the same pile for every pile,
we do not actually have to distinguish between all different composite
restricted blocking patterns. Rather, we can aggregate those co-restricted
composite restricted blocking patterns associated with the same sets of
cards (composite blocking sets) and built from the same number of primi-
tive blocking patterns, because in combining composite restricted blocking
patterns which are not co-restricted, we will never assign different cards
to the same location! We will get back to this in Section 3.4.
It would actually suffice to consider only the parity of the number of
primitive blocking patterns instead of the number itself. We chose to use
the number of primitive blocking patterns in our table for clarity. Note that
it is in fact impossible to have three primitive blocking patterns combine
into one or to have a composite blocking pattern that can be made both
from combining two and from combining four distinct primitive blocking
patterns, so it really does not matter what we choose to do.
We will systematically derive the values in Table 2. The patterns dis-
covered can be used to generate the composite blocking sets much faster
than by brute force. The row concerning zero primitive blocking patterns is
included for completeness. Once we start combining the composite block-
ing sets, we should also take into account the scenarios in which one or
more piles have no primitive blocking patterns in them.
20
```

### PDF page 22

```text
3.3.1 A single primitive blocking pattern
In Table 2, the row concerning a single primitive blocking pattern is com-
puted as described in Section 3.1. Table 3 serves to introduce the notation
used in the remainder of this section and visualizes the structure of com-
positions of primitive blocking sets and the constraints placed upon them.
We will call this a configuration diagram. The letters A represent the cards
of a single primitive blocking set. The∗-symbol indicates which card is the
lock of a this primitive blocking set. We will refer to A∗ as the A-lock. In
the event of multiple letters occupying the same cell, this should be inter-
preted as a set of roles a card plays in different primitive blocking sets.
··· i··· j j+ 1···
♥ A A∗
♦
♣ A
♠ A
Table 3: A visualization of a primitive blocking set. The card values iand
jare constrained by 1 ≤i<j <13.
The contents of the rows with ♥and ♦can be interchanged, while still
signifying a primitive blocking set. Symmetrically, this holds for the con-
tents of the rows with ♣and ♠, and similarly the contents of the bottom
two rows can be interchanged with the contents of the top two rows. Note,
however, that the structure is often invariant under some of these opera-
tions. We need to take this into account when we derive cardinalities from
configuration diagrams.
Figure 7 is a graphical interpretation of the dihedral group D4 [13] as it
relates to the card suits from our perspective. Two configurations are to be
considered equivalent if one can be turned into the other by interchanging
the rows according to the D4 and relabeling the characters of the alphabet
used (i.e., {A,B,...}), but leaving lock annotations untouched.
For each configuration diagram we construct, we are interested in how
many distinct primitive blocking patterns it represents. One of the factors
in this computation is determined by the cardinality of a subgroup of the
D4. In the example depicted in Table 3 this factor is 4.
We can derive the expression for the value in the row with one blocking
21
```

### PDF page 23

```text
Figure 7: The dihedral group D4 as it relates to the permutations of card
suits.
pattern in Table 2 for the pile of height hfrom Table 3:
3! h
4
12
j=2
j−1
i=1
4 = 4 ×3! h
4
12
(j−1) = 1,584 h
4
j=2
The factor 4 is the one mentioned above. The factor 3! follows from the
fact that the three cards indicated with an A can be dealt in any order
before the A-lock. The factor h
4 indicates that the primitive blocking set
can be asigned to any four of the hlocations. Also notice that the number
of composite blocking sets, which we really care about, is much smaller;
in the case of compositions of one primitive blocking pattern it is just 264,
regardless of the pile height.
3.3.2 Compositions of two primitive blocking patterns
There are several ways in which we can combine two primitive blocking
patterns. The structure of the primitive blocking sets of the first possibility
is shown in Table 4. It is the only way to use two primitive blocking pat-
terns while using only five cards. The expression for the contribution of
this composition type to the value of the second row in Table 2 for height
5 ≤his
4! h
5
12
k−1
j−1
k=3
j=2
i=1
4 = 21,120
h
5
where the factor 4 once again follows from interchanging the suits. The
factor 4! stems from the fact that the A,B-lock needs to be dealt later than
the other four cards, but there are no other requirements other than 1 ≤
i < j < k < 13. The contributions of this type of composition to the row
22
```

### PDF page 24

```text
for two primitive blocking patterns in Table 2 for 5 ≤h ≤7 are 21,120;
126,720 and 443,520 respectively.
··· i··· j··· k k+ 1···
♥ A B A∗
,B∗
♦
♣ A,B
♠ A,B
Table 4: A visualization of a composition of two primitive blocking sets
using five cards; 1 ≤i<j <k<13.
··· i··· j··· k k+ 1···
♥ A A∗
♦ B B∗
♣ A,B
♠ A,B
Table 5: A visualization of a composition of two primitive blocking sets
using six cards; 1 ≤i,j <k<13.
Allowing the use of six cards when combining two primitive blocking
patterns gives rise only to compositions of the kind displayed in Table 5.
The expression of the number of composite blocking patterns involved
becomes slightly more complicated. First of all, we need to be careful with
the case where i= j, because it has more automorphisms than the case
where i̸= j. Secondly, the number of ways we can order the cards amongst
themselves is no longer a simple factorial. Either the A-lock or the B-lock
is dealt the last. Therefore, the other lock is dealt earlier, but after the three
cards of the same primitive blocking pattern. This means that we can insert
the remaining card in any of five positions. The resulting expression is
2 ×3! ×5
h
6
12
k=3
k−1
j=2
j−1
i=1
4 +
12
k=2
k−1
j=1
2 = 60,720 h
6
and its contribution to Table 2 is 60,720 and 425,040 for h = 6 and h = 7
respectively.
23
```

### PDF page 25

```text
We can distinguish four different scenarios in which two primitive
blocking patterns are combined and together use seven cards. They are
displayed in Table 6, Table 7, Table 8, and Table 9.
··· i··· j j+ 1··· k k+ 1···
♥ A,B A∗ B∗
♦
♣ A B
♠ A B
Table 6: A visualization of a composition of two primitive blocking sets
using seven cards; 1 ≤i<j <k<13.
··· i··· j j+ 1··· k k+ 1···
♥ A A∗
,B B∗
♦
♣ A B
♠ A B
Table 7: A visualization of a composition of two primitive blocking sets
using seven cards; 1 ≤i<j <k<13.
The structures in Table 6 and Table 7 look nearly identical, but their
subtle difference results in a different contribution. In the one in Table 6
one we can deal either the A-lock or the B-lock last. Whichever is chosen,
the other lock must be dealt earlier and its three associated cards must be
dealt before their lock in any order. Now there are 5 ×6 ways to place
the remaining cards. In contrast, in Table 7 the A-lock cannot be dealt last,
because it would mean that the B-lock is not dealt after all the cards in its
primitive blocking pattern.
Summing these two contributions, we get:
(2 + 1) ×5 ×6 ×3!
12
k=3
k−1
j=2
j−1
i=1
4 = 2,160
12
3 = 475,200
In the computations of the contributions of the last two structures re-
sulting from combining two basic blocking patterns while using seven
24
```

### PDF page 26

```text
cards, shown in Table 8 and Table 9, we should use a factor 8 to acco-
modate the suit symmetry, because all rows are different under all row
interchanges and relabelings.
··· i··· j··· k k+ 1 k+ 2···
♥ A A∗ B
♦ B
♣ B A,B∗
♠ A
Table 8: A visualization of a composition of two primitive blocking sets
using seven cards; 1 ≤i<k<12 and j ≤k.
In Table 8 the B-lock cannot be dealt last, so the A-lock must be. Hence,
the B-lock must be dealt before the A-lock and the other B cards must be
dealt before the B-lock in any of 3! orders. Then there are 5 ×6 pairs of
locations where the remaining A cards can be placed. The contribution
amounts to:
5 ×6 ×3!
11
k=2
k
j=1
k−1
i=1
8 = 633,600
··· i··· j j+ 1··· k k+ 1···
♥ A,B B∗
♦ A
♣ A A∗ B
♠ B
Table 9: A visualization of a composition of two primitive blocking sets
using seven cards; 1 ≤i<j <k−1 <12.
In Table 9, either lock can be dealt last. Once again, the other lock must
be dealt earlier and the cards of the same primitive blocking pattern are
in turn dealt before that in any of 3! different orders. Again, the remaining
two cards can be placed in 5 ×6 ways. This results in a contribution of
2 ×5 ×6 ×3!
12
k=4
k−2
j=2
j−1
i=1
8 = 2,160
12
3 = 475,200,
25
```

### PDF page 27

```text
which gives a total of 2,452,560 composite blocking patterns made from
two primitive blocking patterns and using seven cards.
Because the highest pile has only seven cards, we do not have to count
the number of composite blocking patterns made from two primitive block-
ing patterns and using eight cards.
3.3.3 Compositions of three primitive blocking patterns
There are three basic structures that together describe all ways to com-
bine three primitive blocking patterns using seven cards or less. The first
of these is shown in Table 10. It is an extension of the pattern in Table 4
and uses six cards. There are 12
4 = 495 ways to pick i,j,k and ℓ. The
suit symmetry multiplier is 4. The card that serves as the lock for all three
primtitive blocking patterns has to be dealt last, but other than that there
are no requirements on the order of the cards. The contribution of this type
of composition is
5! h
6
12
4
×4 = 237,600
h
6
which means 237,600 for the pile of height 6 and 1,663,200 for the pile of
height 7.
··· i··· j··· k··· ℓ ℓ+ 1···
♥ A B C A∗
,B∗
,C∗
♦
♣ A,B,C
♠ A,B,C
Table 10: A visualization of a composition of three primitive blocking sets
using six cards; 1 ≤i<j <k<ℓ<13.
The structure displayed in Table 11 is a combination of the structures
in Table 6 and Table 7. There are 12
3 ways to choose i,jand kand the suit
symmetry multiplier is 4. The card that serves as both the B-lock and the
C-lock needs to be dealt last. The A-lock must be dealt after the other cards
marked with A, which can be in any order. The remaining two cards are
both marked with B and C and can be placed in 5 ×6 ways. This results in
a contribution of
5 ×6 ×3! 12
3
×4 = 158,400
26
```

### PDF page 28

```text
··· i··· j j+ 1··· k k+ 1···
♥ A,B A∗
,C B∗
,C∗
♦
♣ A B,C
♠ A B,C
Table 11: A visualization of a composition of three primitive blocking sets
using seven cards; 1 ≤i<j <k<13.
to the number of composite blocking patterns created from three primitive
blocking patterns in the pile of height 7.
··· i··· j··· k··· l l+ 1···
♥ A B A∗
,B∗
♦ C C∗
♣ A,B,C
♠ A,B,C
Table 12: A visualization of a composition of three primitive blocking sets
using seven cards; 1 ≤i<k<l<13 an 1 ≤j <k.
The last composite blocking pattern consisting of three primitive block-
ing patterns is shown in Table 12. If the card that serves as the A-lock and
B-lock is dealt last, then there are 3! ways to order the cards which are
associated with primive blocking pattern C, and there are 5 ×6 ways to
place the remaining two cards. If the card that serves as the C-lock is dealt
last, then there are 4! ways to order the cards associated with primitive
blocking patterns A and B. After that there are six positions to place the
remaining card. This results in the expression
(5 ×6 ×3! + 6 ×4!)
12
l=3
l−1
l−1
k−1
4 = 2,2494,800
k=2
j=1
i=1
which rounds up the corresponding entry in Table 2.
3.3.4 Compositions of four primitive blocking patterns
There is only one type of composition of four primitive blocking sets using
only seven cards, illustrated in Table 13, and this is a further extension of
27
```

### PDF page 29

```text
the structures in Table 4 and Table 11.
... i... j... k... ℓ... m m+ 1...
♥ A B C D A∗
,B∗
,C∗
,D∗
♦
♣ A,B,C,D
♠ A,B,C,D
Table 13: A visualization of the composition of four primitive blocking sets
using seven cards; 1 ≤i<j <k<ℓ<m<13.
The total number of compositions of four primtitive blocking patterns
this structure represents is:
6! 12
5
×4 = 2,280,960
3.4 Composite Restricted Blocking Pattern Aggregation
As was briefly mentioned in Section 3.3, different piles do not share loca-
tions, so when we combine composite restricted blocking patterns that are
not co-restricted, we do not have to take into account the exact locations
the cards are assigned to anymore. We only must not assign the same card
to more than one pile, meaning that it suffices to check that the underlying
composite blocking sets are disjoint for the composite blocking patterns to
be combined.
As a result, we can simply aggregate co-restricted composite restricted
blocking patterns with the same associated composite blocking set which
were constructed from the same number of primitive blocking patterns v
(because we need to know the signs in the inclusion-exclusion formula).
We assign a weight to a tuple consisting of a number indicating the asso-
ciated pile, the composite blocking set, and the number v. Table 14 shows
how much this reduces the amount of data we need to process
This suggests we may need to perform roughtly 2.7 ×1013 evaluations,
which, taking into account that each such evaluation and its subsequent
registration requires some small, but non-negligible amount, of computa-
tion, still takes too long.
28
```

### PDF page 30

```text
Pile height 4 5 6 7
Composite blocking patterns 1,585 29,041 448,801 9,105,361
4,137 Composite blocking sets 265 1,145 21,605
Table 14: The number of composite blocking patterns per pile compared to
the number of composite blocking sets.
3.4.1 Quadruples of tuples
The most straightforward way to combine the tuples formed in the ag-
gregation process is to generate every quadruple consisting of one tuple
associated with every pile pile, and for each such quadruple compute the
contribution to the final answer according to the inclusion-exclusion for-
mula. Each contribution consists of three components:
I The cardinality xof the set of card deals represented by the quadruple.
The four tuples in the quadruple tell us which, but, more importantly,
also how many cards have their location fixed. On the remainder of the
cards there is no restriction. So, given the four composite restricted
blocking sets A, B, C and D, is computed as follows:
x=
(52 −|A|−|B|−|C|−|D|)! if A∩B∩C∩D= ∅
0 otherwise
II The sign of the contribution. This depends on the parity of the sum
of the number of primitive blocking patterns used in the composite
blocking sets in the quadruple.
III A multiplier indicating how many compositions each composite block-
ing set in the quadruple represents as a consequence of aggregation.
3.4.2 Factorial base counter
Because every summand is a multiple of some factorial k! with k∈[30,52]
and these factorials have very large decimal (or even base 264) represen-
tations, which can be relatively cumbersome to add and multiply, we de-
cided to work with a factorial base counter [14].
To avoid having to perform a huge number of carry operations, instead
of limiting the coefficient ck of k! in the factorial base representation to be in
[0,k], we allow ck to be confined to [−263
,263
−1]. Under these constraints
29
```

### PDF page 31

```text
the factorial base numbers do not have a unique representation, but we do
not need them to, as we are not performing comparisons, but only addi-
tions and subtractions. Once a coefficient ck overflows 262 or underflows
−262, we make sure to carry over to or borrow from ck+1.
At the end of the computation we perform a clean-up operation to re-
duce our representation to a canonical factorial base representation start-
ing at the least significant coefficient c30 and moving forward, carrying
over or borrowing from the preceding digits. Because the outcome will be
a positive number, this procedure will terminate.
3.4.3 Bitwise representation and nested loops
If we would nest four loops (one associated with each of the four piles of
height at least 4), and use a 52-bit binary representation of the compos-
ite blocking sets using 64-bit integers, we could check for each quadru-
ple whether the four composite blocking sets are disjoint in a very lim-
ited number of operations. Since we also maintain how many cards each
composite blocking set contains and how many primitive blocking pat-
terns were involved in its creation, and because of the use of a factorial
base counter, we can process each quadruple very fast. However, as was
pointed out in Section 3.4, there are about 2.7 ×1013 such quadruples and
even though we can win some time by backtracking when we encounter
non-empty intersections of sets iterated over by the outer loops, there are
too many quadruples left to evaluate within an acceptable amount of time
on a modern desktop computer.
3.4.4 Exploiting symmetry
A significant factor can be won by exploiting the symmetry among the
suits, which we already encountered while analyzing the primitive block-
ing pattern compositions. Say we iterate over the composite blocking sets
associated with the pile of height 7 in the outer loop. Two composite block-
ing sets that are equivalent under the proper suit transformations (i.e.,
swapping the red suits, swapping the black suits, swapping the black suits
with the red suits and combinations thereof), contribute exactly the same
to the final answer, so we can aggregate them and give a representative of
the equivalence class their total weight.
If we apply this to the pile of height 4, it reduces the computation time
by a factor nearly 4, as each of its composite blocking patterns except for
the empty set shares its equivalence class with three others. If we apply
it to the piles of height 5, 6 and 7, we get closer to the theoretical upper
30
```

### PDF page 32

```text
bound 8 (the number of elements in the D4), because most of their com-
posite blocking sets contain more cards and therefore there are more op-
portunities to break the symmetry. The number of composite blocking sets
to evaluate for the pile of height 7 can be reduced from 21,605 to 4,830,
obtaining a factor slightly less than 4.5.
There is even more to gain if we would be able to perform this re-
duction on combinations of composite blocking sets. As it turns out it is
worthwhile to do this for a combinations of two piles, associated with the
outer two loops. It pays off to precompute these combinations and iter-
ate over them in one outer loop. We will briefly touch upon this again in
Section 3.4.8.
Exploiting the symmetry for combinations of three composite blocking
sets in a way that would still improve the time complexity, seems infeasi-
ble.
3.4.5 Lookup tables and dynamic programming
Of the four nested loops we will be able to eliminate the inner loop by
using a lookup table. We will first analyze how this could work when we
choose the pile of height 4 to be associated with the inner loop. In Sec-
tion 3.4.1 we mentioned the three factors needed to compute the contri-
bution of a quadruple to the final answer. We can easily compute I, the
cardinality of the set of card deals denoted by the quadruple, because it
is almost always the same. Only for the empty set the number of fixed
cards will be the same as for the first three composite blocking sets in the
quadruple; for all others the number of fixed cards will be 4 less. We can
also easily compute II, the sign of the contribution, because the parity of
the number of primitive blocking patterns used is either the same (again,
only in the case of the empty set) or the opposite of the parity for the first
three composite blocking sets in the quadruple. Also III, the number of
compositions of primitive blocking patterns each composite blocking set
associated with the pile of height 4 represents, is always the same, again
with the empty set as a special case.
Finally, we need to compute the number of composite blocking sets that
do not intersect with the union of the first three sets in the quadruple. Re-
calling the structure of the primitive blocking sets, as shown in Table 3, we
present a dynamic programming approach to quickly determine the count.
We split the task up into four parts. In each part we consider a different
suit as the one containing the lock card. Figure 8 illustrates a partial con-
figuration that is represented by some choice of composite blocking sets in
the outer three loops. The red cards are still undetermined and we wonder
31
```

### PDF page 33

```text
Figure 8: A partial configuration which is represented by some choice of
composite blocking sets for the piles of heights 5, 6 and 7.
n 1 2 3 4 5 6 7 8 9 10 11 12 13
♥ x x x x x x
♦ x x
♣ x x x
♠ x x x x
s♥(n) 0 1 1 1 2 2 3 3 3 4 5 5 6
t♥(n) 0 0 0 1 1 1 1 1 4 8 8 13 13
s♦(n) 0 1 2 3 4 4 5 6 6 7 8 9 10
t♦(n) 0 1 1 4 4 4 9 9 15 22 22 31 31
s♣(n) 0 0 1 2 2 3 4 5 6 7 8 9 9
t♣(n) 0 0 1 1 3 3 3 8 14 14 22 22 22
s♠(n) 0 1 1 2 2 3 4 4 5 6 7 8 8
19 19
t♠(n) 0 0 1 1 3 3 3 7 12 12 19 Table 15: A dynamic programming solution for computing the number of
primitive blocking sets that can be made without using the cards marked
by x, applied to the partial configuration of Figure 8.
how many possible choices of composite blocking sets there are left for the
pile with height 4.
In Table 15 we demonstrate the dynamic programming approach to
answering this question. The letter x marks a card that is already in use.
Focusing on the suit of hearts, s♥(n) equals the number of hearts cards
with value smaller than n that have not been used. It is 0 for n = 1 and
for n > 1 it is s♥(n−1) + 1 if the n−1 of hearts has not been used and
32
```

### PDF page 34

```text
s♥(n−1) otherwise. Furthermore, t♥(n) equals the number of primitive
blocking sets for which the value of the lock card is at most n. It is 0 for
n= 1. When neither the ♣(n+ 1) nor the ♠(n+ 1) is used, t♥(n) = t♥(n−
1) + s♥(n), otherwise it is t♥(n−1). We treat the case where no primitive
blocking pattern is present in the pile of height 4 as a special case. We can
fill the table in time linear in the number of cards per suit, whereas a trivial
algorithm would take O(n2) time.
We can preprocess all these values and store them in a large lookup ta-
ble, so we can extract them in O(1) time when we need them. However, we
cannot store the answer for every 52-bit integer, because of the immense
amount of memory required and the enormous time needed to fill the ta-
ble.
Looking back at the dynamic programming solution, notice that to
compute the number of primitive blocking sets with the lock card being
of a specific suit, we only require information about the unused cards of
that suit and the values for which both cards of the other color have not
been used (the bitwise OR proves to be a useful operation here). This means
we can make a 26-bit addressed lookup table instead, at the cost of having
four lookups instead of one. In fact, our lookup table only requires 224 en-
tries, as the presence of the king of our suit and presence of the aces of the
other color are irrelevant.
n 1 2 3 4 5 6 7 8 9 10 11 12 13
u♥(n) 0 0 0 0 0 0 0 0 3 9 9 19 19
u♦(n) 0 0 0 3 3 3 13 13 28 49 49 85 85
u♣(n) 0 0 0 0 1 1 1 11 26 26 54 54 54
17 38 u♠(n) 0 0 0 0 1 1 1 7 17 38 38
Table 16: A dynamic programming solution for computing the number
of composite blocking sets that can be made from 5 cards, applied to the
partial configuration of Figure 8.
Somewhat surprisingly perhaps, we can apply this approach to the pile
of height 5 instead of to the pile of height 4 without complicating matters
much further. We have previously distinguished two structures that apply
to the pile of height 5. The first was the single primitive blocking pattern,
the second was the structure shown in Table 4. We can create a second
24-bit addressed lookup table which is filled similar to the first one. The
only difference is that a function uwill replace the function tand it differs
from tin that we will be adding binomial coefficientss♥(n)
2 instead of just
33
```

### PDF page 35

```text
s♥(n). Table 16 shows ufor the configuration in Figure 8 and Table 15.
We now use twice as much space and precomputation- and lookup
time than when using the pile of height 4, but because the number of com-
posite blocking sets for the pile of height 5 is well over four times larger
than the number of composite blocking sets for the pile of height 4, the
total time spent is still smaller.
We cannot combine the two tables by simply adding the values per-
taining to the different structures, because the values derived from these
tables must be added to neighboring digits in a factorial base, meaning
that the ratio of the contributions depends on their exact position, which
is not known in advance and can in fact vary.
3.4.6 A shattered lookup table
Since the approaches for piles of height 4 and 5 are so similar, a natural
idea would be to extend it to piles of height 6. However, this does not
work, because although the structure from Table 10 can be treated similarly
to the structures we worked with before, we have not found a way to also
accomodate the composite blocking sets displayed in Table 5 in a 24-bit
addressed lookup table.
n 1 2 3 4 5 6 7 8 9 10 11 12 13
v♥♦(n) 0 0 0 3 3 11 11 11 29 57 57 102 102
v♣♠(n) 0 0 1 1 5 14 14 34 64 64 120 120 120
Table 17: A dynamic programming solution for computing the number of
composite blocking sets that can be made from two primitive blocking sets
and using six cards, applied to the partial configuration of Figure 8.
We can use a 36-bit addressed lookup table by keeping track of the
number of unused cards in two suits of the same color separately, similar
to before, this time adding the product of two such values instead of just
one number or a binomial coefficient. We have a function v♥♦(n) (shown
in Table 17), which is 0 for n = 1, and for n > 1 is v♥♦(n−1) in case one
or both of the cards ♣(n+ 1) and ♠(n+ 1) has been used, and v♥♦(n−
1) + s♥(n) ×s♦(n) otherwise. For this table we only need two lookups per
iteration instead of four. However, a table with 236 entries is far too large.
This is where we introduce the concept of a shattered lookup table. A key
insight into its construction is that v♥♦(n) depends solely on v♥♦(n−1),
34
```

### PDF page 36

```text
s♥(n), s♦(n), two bits indicating the absence of the red cards of value n
and a single bit indicating the absence of both black cards of value n+ 1.
We break the lookup table into one heading table and several tailing
tables. The heading table will be indexed by the first q bits of each of the
three rows in the 36-bit addressed table and will contain 23q entries with
values v♥♦(q). We assume it is easy to count bits in a number, for example
by means of a small lookup table. This is how the values of s♥(q+ 1) and
s♦(q+ 1) can be recovered.
There will be (q+1)2 tailing tables with 23×(12−q) entries each, giving the
values v♥♦(12)−v♥♦(q), assuming a different pair of offsets for s♥(q+ 1)
and s♦(q+ 1) in each table.
Figure 9: The graph of f(q) = log2 (23q + 236−3q (q+ 1)2) [15].
What remains to be seen is for which value of q this approach works
best, and whether it works better than having one table with 236 entries
at all. We want to minimize 23q + 236−3q (q+ 1)2 for q ∈[0,12], because we
want to minimize memory usage. Figure 9 quite clearly shows a minimum
near q= 7. So we can create a heading table with 221 entries and 64 tailing
tables with 215 entries each, resulting in 222 entries in total. This is very
reasonable.
35
```

### PDF page 37

```text
3.4.7 Intermezzo: Generalizing shattered lookup tables
The approach with the shattered lookup table can be generalized. We can
make more cuts to make different trade-offs between memory require-
ments and running time, also for larger instances of this problem and for
similar problems. Applying this strategy to eliminate the loop iterating
over the primitive blocking sets associated with the pile of height 7, how-
ever, is complicated by the structure shown in Table 9.
We will take a look at larger instances of the problem, where the deck
has n+ 1 cards per suit. In the case where we want to make only one cut,
we try to minimize fn(q) = 8q +8n−q (q+1)2 for 0 ≤q≤nand q∈N. For our
analysis however, we will consider q∈R. It is easy to see that q∗
n, the value
of qfor which fn(q) is minimal, satisfiesn
2 ≤q∗
n <n. It is also clear that fn
is unimodal on the interval [0,n], i.e., it is monotonically decreasing in the
interval [0,q∗
n] and monotonically increasing in the interval [q∗
n,n]. Hence,
the cut should be made at either ⌊q∗
n⌋or ⌈q∗
n⌉, depending on the values of
fn(⌊q∗
n⌋) and fn(⌈q∗
n⌉).
For simplicity’s sake, let us consider only decks with an odd number
of cards per suit. We compute the following:
fn(n
2 + k) = 23
2 n−3k−2(n2 + (4k+ 4)n+ 4k2 + 8k+ 4 + 26k+2)
fn(n
2 + k+ 1) = 23
2 n−3k−5(n2 + (4k+ 8)n+ 4k2 + 16k+ 16 + 26k+8)
fn(n
2 +k)−fn(n
2 +k+1) = 23
2 n−3k−5(7n2+(28k+4)n+28k2+48k+16−7×26k+5)
Without the strictly positive multiplier 23
2 n−3k−5 of fn(n
2 + k)−fn(n
2 +
k + 1), we are left with a quadratic function of n. For any k its graph is
a parabola which opens upwards and has its focal point below the hori-
zontal axis, since c−
b2+1
= 28k2 + 48k+ 16−7 ×26k+5
(28k+4)2+1
−
4a
4×7 < 0.
Combining these facts, we can conclude that the vertex of each parabola is
also located below the horizontal axis — a somewhat unusual tactic per-
haps, but substituting−b
2a would be unnecessarily messy.
Therefore, for every choice of k ∈N, there exists an nk+1 ∈R, such
that for n > nk+1 we have fn(n
2 + k+ 1) < fn(n
2 + k). The gaps between
the subsequent nk’s become exponentially larger, due to the exponentially
increasing negative offset 7 ×26k+5. Using the abc-formula we find that
36
```

### PDF page 38

```text
12
7
nk+1 corresponds to the positive root:
32
26k+5 +
49−2k−
This means nk equals:
32
26k−1 +
49−2k+
7
Conversely, q∗
nk is roughly¯
nk
¯
2 + log2( ¯ nk )
3 + 1
6 . The first ten transition points are
given in Table 18.
2
k nk
¯
nk
¯
qnk
1 4.000 4 3
2 41.548 42 23
3 356.325 358 182
4 2888.595 2890 1449
5 23160.761 23162 11586
6 185352.086 185354 92683
7 1482896.686 1482898 741456
8 11863267.489 11863268 5931642
9 94906247.910 94906248 47453133
10 759250105.280 759250106 379625063
Table 18: The transition points where f(n
2 + k) is first at least as good as
f(n
2 +k−1) (only taking into account even values of n). The second column
gives the analytical transition point, whereas the third column gives the
practical transition point.
When we aim to make two cuts, at positions qand r(with 2q ≤r), we
want to minimize:
2q + 2r−q (q+ 1)2 + 2n−r(r+ 1)2
As the number of cuts cincreases, it becomes increasingly tricky to solve
the problem with calculus. However, notice that for a given nand a chosen
r we do not have to vary q, since the optimal r for a given q is is known
from the case with one cut.
Based on the intuition that each cut cuti with i ∈ [0,cuts−1] in an
optimal configuration will obey |cuti−i n
cuts+1 |= O(log(n)), we present an
algorithm to compute the optimal cuts with time complexity O(log(n)cuts ).
Pseudocode for the algorithm is shown in Figure 10.
37
```

### PDF page 39

```text
6 t h i s s i z e >= 0 && t h i s s i z e ∗ chunks <= n ;
7 t h i s s i z e += d e l t a ) {
8 t h i s p a i r = solve ( n− t h i s s i z e ,
9 chunks− 1 ,
t h i s s i z e ,
memory + 8∗∗t h i s s i z e
∗ ( n− t h i s s i z e + 1) ∗∗2 ,
1 pair solve ( n , chunks , l a s t s i z e , memory , d e l t a ) {
2 i f ( chunks == 1)
3 return ( 1 , [ n ] )
4 b e s t p a i r = ( i n f i n i t y , [ ] )
5 for ( t h i s s i z e = l a s t s i z e ;
10 11 12 13 14 15 16 17 18 19 }
20 }
21 22 }
23 1)
i f ( t h i s p a i r . f i r s t < b e s t p a i r . f i r s t ) {
b e s t p a i r = t h i s p a i r
b e s t p a i r . second += [ t h i s s i z e ]
} e lse {
break
return b e s t p a i r
pair s o l u t i o n = solve (N, CHUNKS, N / CHUNKS, 0 ,−1)
Figure 10: A pseudo code implementation of an algorithm to compute op-
timal cuts for the generalized shattered lookup table problem.
For smaller values of nwe can speed this up slightly by computing the
solution analytically when chunks equals 2 (assuming the mathematical
operations can be performed in constant time for such values of n).
Table 19 lists some optimal solutions for two, three and four cuts, which
were computed with the given algorithm. Keeping in mind that the algo-
rithm works with values that are relatively close to each other, a useful
trick to make it work for relatively large values of n without having to
work with very long numbers, is to represent the summands as pairs (a,b),
indicating a×2b
.
3.4.8 Final notes on exploiting symmetry
Because we replaced the loop associated with the pile of height 6 by a
number of lookups, it makes sense to apply the exploitation of the suit
symmetry as described in Section 3.4.4 to a combination of the piles of
height 5 and height 7.
When we do not exploit any symmetry, there are 10,665,653 combina-
tions of one set associated with the pile of height 5 and one set associated
38
```

### PDF page 40

```text
n 2 cuts 3 cuts 4 cuts
3 [2,1,0]
4 [2,1,1] [2,1,1,0]
5 [3,1,1] [2,1,1,1] [2,1,1,1,0]
6 [3,2,1] [3,1,1,1] [3,1,1,1,0]
7 [3,2,2] [3,2,1,1] [3,1,1,1,1]
8 [4,2,2] [3,2,2,1] [3,2,1,1,1]
9 [4,3,2] [4,2,2,1] [3,2,2,1,1]
10 [5,3,2] [4,3,2,1] [4,2,2,1,1]
11 [5,3,3] [4,3,2,2] [4,3,2,1,1]
12 [5,4,3] [5,3,2,2] [4,3,2,2,1]
13 [6,4,3] [5,3,3,2] [4,3,2,2,2]
14 [6,4,4] [5,4,3,2] [5,3,2,2,2]
15 [6,5,4] [5,4,3,3] [5,3,3,2,2]
16 [7,5,4] [6,4,3,3] [5,4,3,2,2]
17 [7,5,5] [6,4,4,3] [5,4,3,3,2]
18 [7,6,5] [6,5,4,3] [6,4,3,3,2]
19 [8,6,5] [7,5,4,3] [6,4,3,3,3]
20 [8,6,6] [7,5,4,4] [6,4,4,3,3]
100 [36,32,32] [28,25,24,23] [23,20,19,19,19]
1000 [337,332,331] [255,249,248,248] [205,200,199,198,198]
Table 19: Optimal solutions for various larger instances of the shattered
lookup table problem.
with the pile of height 7 such that the sets are disjoint. This is significantly
less than 1,145 ×21,605 = 24,737,725.
If we only exploit symmetry in the pile of height 7, there are 2,527,065
combinations of one set associated with the pile of height 5 and one set
associated with the pile of height 7 such that the sets are disjoint.
When we exploit the symmetry of both the pile of height 5 and the pile
of height 7 together (it does not matter for the outcome whether we first
exploit the symmetry in the pile of height 7 or not, but it will be faster if
we do), and disregard the combinations where the sets are not disjoint,
1,700,305 cases reamin to iterate over. This means we can attribute a factor
10, 665, 653
1, 700, 305 ≈6.27 of reduction in the number of cases to our exploitation of
suit symmetry.
39
```

### PDF page 41

```text
3.5 Results
The final implementation of our algorithm ran in under 85 seconds on
a regular desktop computer. It determined the number of distinct type I
blocked Klondike Solitaire deals to be a number with 66 decimal digits:
952,628,704,275,585,774,781,929,328,392,634,005,105,712,508,518,715,873,034,240,000,000
When we divide this number by 52!, we get the reduced fraction
21649537217724514188460974601
1833045925909798584842457600000
as the fraction of type I blocked Klondike Solitaire deals. This amounts to
approximately 1.181069% of all games.
3.6 Extension to Type II
A natural extension of blocked Klondike Solitaire deals of type I, is the
class of configurations in which one or more primitive blocking sets are
intertwined in a way that a lock card might not necessary lock up the re-
mainder of its primitive blocking set within its own pile, but the lock cards
together do lock up the remainders of their primitive blocking sets. We call
these configurations deals of type II.
Figure 11: A schematic representation of a Klondike Solitaire deal of type
II. Three of the primitive blocking sets that can be discerned in this image
are {♠3,♠5,♥6,♦6}, {♠3,♠9,♥10,♦10}and {♥6,♥7,♣8,♠8}.
A schematic representation of a set of deals that are all of type II, but
not all of which are of type I, is displayed in Figure 11. From the posi-
tioning of the primitive blocking sets {♠3,♠5,♥6,♦6}, {♠3,♠9,♥10,♦10}
40
```

### PDF page 42

```text
and {♥6,♥7,♣8,♠8}alone we can conclude that these games cannot be
finished, even though no primitive blocking pattern might be present: The
♠5 cannot be played until one of the cards ♥6 and ♠3 has been freed, but
for this to be possible the ♥7 or the ♠9 must be moved. The ♥7 cannot be
moved until the ♣8 been freed, which requires the ♠5 to be moved first.
The ♠9 cannot be moved until one of the red tens has been freed, but for
this to be possible either the ♥7 or the ♠5 must be moved. From this it
follows that neither the ♥7, nor the ♠5, nor the ♠9 can ever be moved.
We were not able to compute the exact number of card deals of type
II, but we performed a Monte Carlo simulation with a Mersenne Twister
as a random number generator to get a good estimate [16]. In 10,000,000
random deals, 217,328 were of type II, meaning about 2.17% of all deals is
expected to contain a conflict of this kind.
The accuracy of this result depends on the quality of the random num-
ber generator used. We do think that the result reported is accurate, be-
cause the same random number generator and a comparable simulation
were applied to estimate the percentage of card deals of type I and in this
case even much higher accuracy was attained.
41
```

### PDF page 43

```text
4 Klondike Draws
One of the constraining factors when playing Klondike Solitaire is the fact
that not every card from the stock is available at all times, as explained in
Section 1.1. In order to get some measure of the limitations imposed by the
stock, we study how many permutations of cards can be extracted from it.
As an aside, if we would want to quickly determine whether a given
permutation can be extracted from a given stock, we could quite efficiently
do so, using a Fenwick Tree [17]. We first relabel the elements, such that the
cards in the stock are labeled from 1 through nin increasing order, in O(n)
time and space. Next, we build a Fenwick tree on top of a frequency table
with nentries of 1, in O(nlog n) time. We keep track of the position of the
last extracted card and repeatedly check whether the next card is either a
multiple of three cards ahead or a multiple of three cards from the start. If
not, we can conclude that the given permutation cannot be extracted from
the stock. If we can process the entire permutation, it can be extracted.
All lookups of cumulatives and all card removals can be done in O(log n)
time per query, hence, the worst running case time of this algorithm is
O(nlog n).
4.1 Deriving a Recurrence
Let A(n,k) denote the number of permutations of n>0 elements that can
be generated by a so-called Klondike draw of degree k. Such a draw works
in the following way. Consider the sequence of elements 1,2,...,n, in this
order. We proceed in stages. In every stage at least one number must be
removed from the sequence, and appended to the permutation that is be-
ing generated; the process terminates as soon as the last element has been
removed. In a single stage one is allowed to do the following. First choose
tintegers 1 ≤i1 <i2 <...<it ≤nj , the number of remaining elements (n
in the first stage), all a multiple of k — except for possibly it, which may
also be equal to nj , even if this is not a multiple of k. These numbers in-
dicate the available positions. The number tis called the stage rank. Then
pick the i1-th element, the (i1−1)-th element, the (i1−2)-th element,... (as
many as one wants, and of course when still available), the i2-th element,
the (i2−1)-th element, the (i2−2)-th element,... and so on. Of course,
elements can only be removed once. Note that when taking, e.g., the i2-th
element, still the original numbering is used.
If during a stage kelements are removed, it is possible to abandon fur-
ther actions in this stage, and postpone them to the next. The last few ele-
ments play a somewhat special role: indeed, they can always be removed.
42
```

### PDF page 44

```text
We clearly have A(n,1) = n!, and A(n,k) = 1 if k ≥n. So we may
assume 1 <k<n. Now the main recursion is:
A(n,k) = ⌈n/k⌉A(n−1,k) + ⌈n/k⌉
2
A(n−2,k)
+ ⌈n/k⌉+ 1
3
A(n−3,k) +...+ ⌈n/k⌉+ k−2
k A(n−k,k)
= A(n−1,k) +
k
i=1
⌈n/k⌉+ i−2
i
A(n−i,k).
This can be proven as follows. Suppose that during the first stage we
want to pick r ≤k elements. We do not want to choose the last element
(except perhaps when r = 1, which easily leads to the first term from
the righthand side of the formula). We then have to choose r elements
from the ⌈n/k⌉−1 elements at the available positions (thereby excluding
the last one, if nhappens to be a multiple of k), with repetitions allowed.
A repetition simulates the case where we keep on taking elements before
proceeding to the next multiple of k. This gives a contribution of ⌈n/k⌉+r−2
r
possibilities. Now one can start a new stage with relements less. Note that
usage of the last element is fully taken care of in the situation with r= 1.
n\k 1 2 3 4 5 6 7 8 9 10 11 12
1 1 1 1 1 1 1 1 1 1 1 1 1
2 2 1 1 1 1 1 1 1 1 1 1 1
3 6 3 1 1 1 1 1 1 1 1 1 1
4 24 7 4 1 1 1 1 1 1 1 1 1
5 120 30 10 5 1 1 1 1 1 1 1 1
6 720 111 25 13 6 1 1 1 1 1 1 1
7 5,040 624 121 33 16 7 1 1 1 1 1 1
8 40,320 3,162 478 85 41 19 8 1 1 1 1 1
9 362,880 22,050 1,897 431 106 49 22 9 1 1 1 1
10 3,628,800 141,870 11,666 1,745 276 127 57 25 10 1 1 1
11 39,916,800 1,181,970 62,826 7,033 1,426 331 148 65 28 11 1 1
12 479,001,600 9,219,870 340,270 28,483 5,831 865 386 169 73 31 12 1
Table 20: The number of Klondike draws A(n,k) of length nand degree k
for 1 ≤n,k≤12. The values for k<n≤2kare underlined for clarity.
Table 20 lists the number of Klondike draws of length nand degree k
43
```

### PDF page 45

```text
for small values of n and k as computed using the recurrence given and
confirmed with a brute force solution.
A number of special interest is A(24,3), since there are 24 cards in
the stock in Klondike Solitaire, and under standard rules stock cards are
flipped over per three. We found that A(24,3) = 104,244,550,813,439,400,
meaning roughly only one in six million permutations of 24 cards can ac-
tually be extracted from the stock.
4.2 Klondike Draws with k<n≤2k
Some patterns emerge within the number jumble of Table 20. For example,
when we consider the values A(n,k) for k <n≤2k(underlined in the ta-
ble), we have ⌈n
k ⌉= 2 and ⌈n
k ⌉+i−2
i = 1, so our recursive formula reduces
to:
A(n,k) = A(n−1,k) +
k
A(n−i,k),
i=1
from which we can derive the following expression in which the number
of terms is independent from nand k:
A(n,k)−A(n−1,k) = 2A(n−1,k)−A(n−2,k)−A(n−1−k,k),
or equivalently,
A(n,k) = 3A(n−1,k)−A(n−2,k)−1,
since A(n,k) = 1 for n≤k.
By solving this linear homogeneous recurrence relation with constant
coefficients [18], for the same restrictions on n and k we can derive the
following closed formula:
A(n,k) = 1 + k×(3 + √5)n−k
−(3−√5)n−k
2n−k√5
This formula reveals a close relationship with the Fibonacci numbers [19].
In fact, 3+√5
2 = (1+√5
2 )2
= φ2 and 3−√5
2 = (1−√5
2 )2 = (1−φ)2, and using Bi-
net’s formula [20] we obtain:
A(n,k) = 1 + k×F2n−2k,
and
A(n,k)−A(n−1,k−1) = F2n−2k,
as can easily be verified along the backslash diagonals within the under-
lined area.
44
```

### PDF page 46

```text
4.3 Klondike Draws with 2k<n≤3k
For 2k<n≤3kwe know ⌈n
k ⌉= 3 and our main recursion reduces to
A(n,k) = A(n−1,k) +
k
(i+ 1)A(n−i,k).
i=1
Now let us consider what A(n,k)−2A(n−1,k) + A(n−2,k) looks like.
In the notation below we will keep k fixed and denote A(n,k) as an for
simplicity.
3an−1 +3an−2 +4an−3 +5an−4. . . +(k + 1)an−k
−3an−2−3an−3−4an−4. . .
−kan−k−(k + 1)an−k−1
−3an−2−3an−3−4an−4. . .
−kan−k−(k + 1)an−k−1
+3an−3 +3an−4. . . +(k−1)an−k +kan−k−1 +(k + 1)an−k−2
3an−1−3an−2 +an−3 +0. . . +0−(k + 2)an−k−1 +(k + 1)an−k−2
Conveniently, all but five terms are eliminated in the addition, which
leads us to the following alternative recursive formula for 2k<n≤3k:
A(n,k) = 5A(n−1,k)−4A(n−2,k) + A(n−3,k)
−(k+ 2)A(n−k−1,k) + (k+ 1)A(n−k−2,k).
4.4 Pfaff-Fuss-Catalan as a Lower Bound
In Section 1.1 we mentioned that Klondike Solitaire is sometimes played
with at most three passes through the stock. The number of permutations
that can be extracted from the stock in a constant number of passes cis a
lower bound Lc(n,k) on A(n,k). We will consider the case c= 1.
While extracting a permutation we repeatedly flip kcards, a total num-
ber of f= ⌈n
k ⌉times and we draw a single card n times, under the con-
straint that at no point we are to have drawn more cards than we have
flipped over. It is easily verified that L1(n,k) = L1(k⌈n
k ⌉,k), so for simplic-
ity we will assume that kdivides n.
The situation can be thought of as a rectangular lattice grid on which
we want to count the number of monotonic paths from (0,0) to (n,f), per-
mitting only steps (0,1) and (1,0), and without ever going below the line
x
y=
k.
We mimic a proof for the formula for Catalan numbers as it is found
on Wikipedia [21]. Figure 12 serves to help illustrate the proof.
45
```

### PDF page 47

```text
Figure 12: A monotonic path with exceedence 7 (above) and a modified
version with exceedence 6.
We define the exceedence of a monotonic path as the number of horizon-
tal edges which lie below the line y=
x
k . In the upper path in Figure 12 this
is 7.
In case a monotonic path does not have exceedence 0, there must exist
a rightmost horizontal edge starting at the line y=
x
k . We can swap the
part of the path before and after this edge to end up with a path that has
exceedence one lower than the original path had. After all, neither the part
that was originally on the left, nor the part that was originally on the right
had its exceedence altered, but the edge connecting them now has to be
located above the line y=
x
k , while earlier it was not.
The process is reversible: For any path P with exceedence less than n
there exists a unique path that results in P when the procedure described
is used. In the reverse procedure we identify the first horizontal edge that
ends at the line y=
x
k.
From this we can conclude that the number of monotonic paths from
(0,0) to (n,k) with exceedence 0 equals the number of monotonic paths
46
```

### PDF page 48

```text
1
from (0,0) to (n,k) of any exceedence ewith 1 ≤e≤n. As a consequence,
the number of paths with exceedence 0 must equal the total number of
monotonic paths, which isn+f
n , divided by n+ 1. Hence,
L1(n,k) = (k+ 1)⌈n
k ⌉
k⌈n
k ⌉
n+ 1.
In particular, L1(24,3) = 420,732.
For k = 1, L1(n,k) = Cn, the n-th Catalan number. More generally,
Lc(n,k) equals Ck+1
⌈n
k ⌉, where Cm
n is the Pfaff-Fuss-Catalan sequence [22, 23].
47
```

### PDF page 49

```text
5 Counting No-braider Klondike Solitaire Deals
Consider a rule set for Klondike Solitaire in which all cards in the stock are
accessible. We will define a no-braider deal as a game which can be won by
solely applying the following two types of moves:
1) Moving a card from the stock directly to the foundation.
2) Moving a card from the table directly to the foundation.
In this section we will lay the groundwork for computing the total
number of no-braider deals.
5.1 Braid Numbers
Let the vectors v,w ∈N∗each represent a row of piles of cards such that
the height of the p-th pile within a row is specified by the p-th element of its
respective vector. All cards within a row are considered distinct. Starting
with such a row of card piles, we will repeatedly take one of the topmost
cards and move it onto a (possibly empty) pile in a second row of cards.
We continue until all cards have been moved to the second row. Now, the
braid number B(v →w) denotes the number of possible configurations of
the second row of cards when the heights of the piles in the first row are
described by vand the heights of the piles in the second row are described
by w.
Figure 13 gives an example of the way cards could be moved from a
configuration with pile heights described by the vector (3,1,2) to a config-
uration with pile heights described by the vector (2,1,3). The configura-
tion in the lower right is counted only once in B((3,1,2) →(2,1,3)), even
though there are multiple ways to reach it. For example, the yellow card
could already have been moved in the third step.
The braid number is invariant under permutations of the vectors vand
w, that is to say that:
∀π1,π2 : B(v→w) = B(π1(v) →π2(w))
Insertion or removal of zero elements does not alter a vector’s behavior
when it comes to braid numbers. Furthermore — and this is not entirely
trivial — the function is symmetric, meaning that B(v →w) = B(w →v)
for all v, w.
For v = (v1,...,vk) ∈Nk, we say that |v|= k and we define ||v||as
|v|
i=1 vi. We define the braid number of vand wto be zero whenever ||v||̸=
||w||.
48
```

### PDF page 50

```text
Figure 13: An example of the way cards could be moved from a configu-
ration with pile heights described by the vector (3,1,2) to a configuration
with pile heights described by the vector (2,1,3).
When v = (v1) ∈ N1
, w = (w1,...,wk) ∈ Nk and ||v||= ||w||, the
49
```

### PDF page 51

```text
following holds true:
B(v→w) = v1!
w1!···wk!
In particular, when |v|= |w|= 1, we have B(v→w) = 1.
If vi ≤1 for 1 ≤i ≤|v|, then for any w with ||v||= ||w||we have
B(v→w) = ||v||!.
5.2 A Recursive Definition for B((h1,h2) →(ℓ1,ℓ2))
Let h1 + h2 = ℓ1 + ℓ2, where h1 and h2 are the heights of the original left
and right pile, and ℓ1 and ℓ2 are the heights of the final left and right pile.
Then we have
B((h1,h2) →(ℓ1,ℓ2)) =
min(h1,ℓ1−1)
min(h1−t,ℓ2)
t=0
s=0
s+ t−1
s
B((h1−s−t,h2−1) →(ℓ1−t−1,ℓ2−s))
+ similar with ℓ1 ↔ℓ2
This corresponds with the following. The top element of the original right
pile, let us call this element b, ends up in either the final left or right pile.
Assume it does so in the left pile, with t elements (that must have come
from the original left pile) underneath. Clearly, this tmust satisfy t≤ℓ1−1
and t ≤h1. The element immediately underneath b (if any) determines
how many elements from the original left pile certainly must have already
been moved: suppose there are t+sof these, where selements have moved
to the right pile. Clearly, s ≤ℓ2 and s ≤h1−t. Note that more elements
from the original left pile could have been moved to the final right pile, but
one can assume that bwas moved now. (If t = 0, the formula also holds,
though other splits might be also usable.)
The t+ selements could have produceds+t−1
s different pairs of piles,
having the topmost extra element in the final left pile fixed. It remains to
count how many possibilities there are for moving piles of heights h1−t−s
and h2−1 to piles of heights ℓ1−t−1 and ℓ2−s.
If ℓ1 = ℓ2, we simply get a factor 2.
5.3 A Case Study: B((h,h) →(h,h))
We can compute an upper bound U(h) for B((h,h) → (h,h)) relatively
easily: For 0 ≤i ≤h, there are h
i ways to choose i cards from the left-
most starting pile to move to the leftmost end pile. There are also h
i ways
50
```

### PDF page 52

```text
to choose i out of h locations in the leftmost end pile where these cards
should be put. The order in which these icards are mapped to their new lo-
cations is fixed. Now there are h
h−i ways to choose which h−icards from
the rightmost starting pile will end up in the leftmost end pile and h
h−i
ways to choose which positions the remaining cards from the leftmost
starting pile will occupy in the rightmost end pile. Since everything else
is fixed and h
i
=
h
h−i , we end up with U(h) = h
i=0
h
i
4. This happens
to equal 4F3(−h,−h,−h,−h; 1,1,1; 1), where pFq (a1,...,ap; b1,...,bq ; z) is
the generalized hypergeometric series [15].
Figure 14: Neither of the configurations on the right-hand side can be ob-
tained from the configuration on the left.
To demonstrate that U(h) is not in general a tight upper bound, we take
h = 2 and compare U(2) = 18 with B((2,2) →(2,2)) = 16. The difference
stems from two unattainable end configurations (one of these is depicted
at the bottom in Figure 14, the other is that same configuration with the
piles interchanged) in which the original top cards are on top once again,
each covering the card that was originally at the bottom of the other pile.
The resulting cyclic dependency among the four cards violates the rules
set forth, but is not properly taken care of in the computation of U(h).
A preliminary brute force algorithm unexpectedly brought a very inter-
esting pattern to light regarding the braid numbers of the form B((h,h) →
(h,h)). We then conjectured:
B((h,h) →(h,h)) = Q2
h ×2h mod 2
,
where Qh ∈N depends on h.
The speculated pattern in the values for B((h,h) →(h,h)) might hint at
the existence of a manageable underlying structure yet to be discovered.
51
```

### PDF page 53

```text
5.4 An Algorithm to Compute Braid Numbers
So far, we have not found an elegant one-line recursive definition of B(v,w),
but we did design a fairly efficient algorithm based on memoization [24],
which greatly outperforms any brute force solution.
We use a double-layered depth-first search, with both layers governed
by the same function and subject to the same memoization method. In the
outer layer we choose any possible matching between cards currently on
top in the starting piles and bottommost open locations in the end piles,
including a possibility for each top card to not be matched in this round.
It is imperative that the current top cards in the starting piles, if they are
not matched during this round, will not be matched with the locations
currently available to them in any subsequent round, so as to not double
count configurations. To enable this, each of the starting piles has a bit vec-
tor of length equal to the number of end piles associated with it. Whenever
a location in one of the end piles is filled, the forbid-bits associated with
this end pile must be set to 0, and whenever a card currently at the top on
one of the starting piles is matched, all of its forbid-bits must be set to 0.
We have to be careful what we manage on the fly and what we manage at
the end of a round (we do not want to match two cards to locations within
the same pile within a single round, for example).
In the inner layer of our nested depth-first search we iterate — by
means of recursion — over the starting piles in order to match the card
currently on top to an available location in the end piles. We also allow
this card to not be matched. To avoid an infinite recursion, we disallow
rounds in which no cards are matched at all.
We drastically reduced the number of states to store and at the same
time significantly increased the probability of a hit during a memoization
lookup by normalizing states before starting a new matching, by sorting
the starting piles and their associated forbid-bits.
Table 21 shows braid numbers for configurations in which cards are
moved from two piles of height hto two new piles of height h. The right-
most column serves to reveal the pattern we discovered. The sequence has
been added to The Online Encyclopedia of Integer Sequences as A214623 [25].
Table 22 lists braid numbers for configurations in which there are three,
four and five piles of height hrespectively, from which cards are moved to
equally many end piles.
52
```

### PDF page 54

```text
h B((h,h) →(h,h)) 2h (mod 2) ×Q2
h
0 1 12
1 2 2 ×12
2 16 42
3 128 2 ×82
4 1,156 342
5 10,952 2 ×742
6 107,584 3282
7 1,083,392 2 ×7362
8 11,115,556 3,3342
9 115,702,472 2 ×7,6062
10 1,218,289,216 34,9042
11 12,948,910,592 2 ×80,4642
12 138,708,574,096 372,4362
13 1,495,661,223,968 2 ×864,7722
14 16,218,468,710,656 4,027,2162
15 176,727,219,273,728 2 ×9,400,1922
16 1,933,956,651,447,076 43,976,7742
17 21,243,204,576,601,928 2 ×103,061,1582
18 234,121,111,199,439,424 483,860,6322
19 2,587,943,032,046,002,688 2 ×1,137,528,6882
Table 21: Braid numbers for a configuration in which cards are moved from
two piles of height hto two new piles of height h.
5.5 A Graph Theoretical Interpretation
We can look at braid numbers in a graph theoretical context. Let G1(V,E1)
and G2(V,E2) be forests of directed paths. A directed edge connects two
vertices which represent cards placed directly on top of one another in
the associated pile configuration. The direction of an edge signifies which
of the two cards must be moved first or must have been moved first. In
such a graph no vertex has an in-degree greater than one and no vertex
has an out-degree greater than one. Let g1 and g2 be vectors describing the
sizes of all weakly connected components in G1 and G2 respectively. Now
B(g1 →g2) is the number of permutations π ∈S|V ] for which the graph
G(V,E1 ∪π(E2)) is acyclic, where with π(E2) we mean a permutation with
respect to the underlying set of vertices V of the edge set E2.
53
```

### PDF page 55

```text
h B((h)3 →(h)3) B((h)4 →(h)4) B((h)5 →(h)5)
0 1 1 1
1 6 24 120
2 504 28,800 2,620,800
3 64,368 82,616,832 218,855,808,000
4 9,565,236 300,656,771,136
5 1,578,247,416 1,269,472,367,476,224
6 279,149,957,136
7 51,881,173,065,504
8 10,006,245,507,952,548
9 1,986,141,173,595,782,616
Table 22: Braid numbers for a configuration in which cards are moved from
three, four or five piles of height hto as many new piles of height h.
5.6 A Case Study: B(R2,k →R2,k)
Orthogonal to the case where there are two piles of height h as both the
starting and the end configuration, we examine the situation where we
start and end with k piles of height 2. We use the notation R2,k for the
corresponding vector.
The graph theoretical analogue of B(R2,k → R2,k) is the number of
acyclic compositions (as defined above) of two 1-regular directed graphs
on 2k vertices. It is easily demonstrated that each such composition is 2-
regular.
Every 2-regular directed multigraph has an underlying undirected multi-
graph which is composed entirely of disconnected cycles [26]. This fact
allows us to derive a recursive formula for B(R2,k →R2,k).
Figure 15 and 16 show a 1-regular directed graph and a 2-regular di-
rected multigraph respectively. The underlying undirected multigraph of
the graph in Figure 16 consists of three cycles. The multigraph itself how-
ever, contains only a single cycle (ABEF). This cycle should be interpreted
as a cyclic dependency, constituting an invalid composition in the context
of braid numbers.
5.6.1 A recursive definition and a closed formula
Let ak = B(R2,k →R2,k). We can derive a recursive definition for ak as
follows. The leftmost end pile corresponds to a directed edge in the com-
54
```

### PDF page 56

```text
Figure 15: A 1-regular directed graph.
Figure 16: A 2-regular directed multigraph. The double arrow from node
C to node Ddenotes a double edge.
posite graph; it is part of a cycle in the underlying undirected graph which
further consists of k
t edges corresponding to tout of kstarting piles and
k−1
t−1 edges corresponding to other end piles, for 1 ≤t ≤n. These edges
alternate in the cycle. Each of the remaining 2t−1 edges can be oriented in
any of two directions with regard to the edge corresponding to the leftmost
end pile. Exactly one of the 22t−1 possibilities forms a cycle in the directed
55
```

### PDF page 57

```text
graph. Each of the other (undirected) cycles can be completed to form an
acyclic composition in exactly ak−t ways. This derivation translates to the
following recursive definition of ak:
ak =
k
t=1
ak−t
k
t
k−1
t−1
t!(t−1)!(22t−1
−1)
=
k−1
t=0
at
k
k−t
k−1
k−t−1 (k−t)!(k−t−1)!(22k−2t−1
−1)
=
k−1
at
t=0
k!(k−1)!
(t!)2 (22k−2t−1
−1)
=
k−1
t=0
at
k!!(k−1)!!
(t!!)2−
k!(k−1)!
(t!)2
k B(R2,k →R2,k)
0 1
1 1
2 16
3 504
4 28,800
5 2,620,800
6 348,364,800
7 63,707,212,800
8 15,343,379,251,200
9 4,707,627,724,800,000
10 1,792,664,637,603,840,000
Table 23: Braid numbers for a configuration in which cards are moved from
kpiles of height 2 to knew piles of height 2.
Table 23 shows the braid numbers B(R2,k →R2,k) for 0 ≤k ≤10. We
confirmed the values for k ≤7 with an implementation of the algorithm
described in Section 5.4. The sequence has been added to The Online En-
cyclopedia of Integer Sequences as A214624 [25].
Without further proof, we conjecture that for k ≥ 0, the following
closed formula holds:
ak = (2k)!−k2(2k−2)! = 3k−2
4k−2(2k)!
56
```

### PDF page 58

```text
This expression suggests a much more straightforward alternative re-
currence relation, in which a0 = 1 and for k>0:
2k(2k−3)(3k−2)
ak = ak−1
3k−5
5.7 Application of Braid Numbers to Klondike Solitaire
The total number of no-braider Klondike Solitaire games when all the
cards in the stock are accessible (which is in essence equivalent with flip-
ping one card each time) equals the braid number B(((1)25
,2,3,4,5,6,7) →
(13,13,13,13)). Here we consider every stock card to be its own pile of
height 1. Computing this number with the algorithm from Section 5.4
would take too long. Luckily, we can treat piles with a single card in a
special way. A starting pile of height 1 cannot be part of any cyclic depen-
dency, hence in our case 25 of the cards involved can be placed anywhere
without constraints.
Let us first examine a smaller instance of the problem. Say we want
to compute B((1,4,5) →(4,6)). The card in the pile of height 1 could go
to any of the four locations in the leftmost end pile, where it can be inter-
leaved with any of B((4,5) →(3,6)) = 2,544 possibilities to assign the nine
remaining cards. Similarly, it can go to any of the six locations in the right-
most pile, where it can be interleaved with any of B((4,5) →(4,5)) = 3,316
possibilities to assign the remaining cards. Hence, B((1,4,5) →(4,6)) =
4 ×2,544 + 6 ×3,316 = 30,072.
When kpiles with a single card are in play, we have to incorporate the
number of ways in which these kcards can be ordered, k!, and the number
of ways a single such ordered collection of k cards can be assigned to k
locations within the end piles. If there are ℓend piles, this can be expressed
as a summation over products of ℓbinomial coefficients; in our case:
B(((1)25
,2,3,4,5,6,7) →(13,13,13,13)) =
25! ×
a+b+c+d=25
13
a
13
b
13
c
13
d
×
B((2,3,4,5,6,7) →(13−a,13−b,13−c,13−d))
This is still somewhat computationally intensive, but using the frame-
work for which we laid the groundwork in this section and the current
state of our understanding of braid numbers, we think we will very soon
be able to compute the number of no-braider Klondike Solitaire games
within a reasonable amount of time.
57
```

### PDF page 59

```text
As an aside, we know from Section 4 how many permutations can be
extracted from the stock when flipping cards per three, and it is tempting
to consequently consider the stock as one pile of height 24 and multiply
the result by A(24,3). However, this logic is flawed, just like it is incorrect
to state that B((2,2) →(2,2)) = 16 = 4
2 ×B((4) →(2,2)) = 6 ×6.
58
```

### PDF page 60

```text
6 Conclusions and Future Research
We studied the problem of counting the number of Klondike Solitaire
games in which not a single move can be made, and we presented a very
fast Dynamic Programming solution. The algorithm scales linearly1 in each
of the relevant dimensions (the number of piles on the tableau, the num-
ber of initially accessible cards in the stock and the number of cards per
suit). The value computed for the standard version of Klondike Solitaire
corroborates the result reported by Donkersteeg and Kosters [10].
We successfully extended this approach to also count the number of
games in which a single ace can be moved to the foundation, after which
no moves are possible anymore. Together these values constitute a lower
bound of 0.3645...% of games that cannot be won. It would be feasible to
push this limit further in a similar vein in future research.
We established another lower bound on the number of Klondike Soli-
taire games that cannot be won by counting starting configurations in
which cards within a single pile on the tableau can together block the
game. We carefully examined the structures and symmetries of compos-
ite blocking patterns, and subsequently designed an algorithm based on:
the Inclusion-Exclusion principle; lookup tables; a factorial base counter
with large-range coefficients; and a hybrid between a lookup table and a
dynamic programming approach. The latter, coined a shattered lookup ta-
ble, can be generalized to accomodate trade-offs between space and time
under comparable circumstances, and we briefly touched upon this. We
believe that the lower bound of 1.181069...% of games that cannot be
completed because of these blocking configurations might correspond to
the claim by Yan, Diaconis, Rusmevichientong and Van Roy [6] of an up-
per bound of solvable games of 98.81%, for which a faster implementation
was under way as of 2005.
We established that 2.17...% of all games2 cannot be completed due
to the presence of one or more blocking patterns from a wider class of
blocking patterns, including patterns that span more than a single pile,
by means of a Monte Carlo simulation with a Mersenne Twister [16] as
random number generator. We tried to devise an algorithm to compute
this number exactly within a reasonable amount of time, but we were as
of yet unsuccessful.
We investigated to what extent the stock limits game play in Klondike
Solitaire. We discovered a recurrence for the number of permutations that
1Not taking into account the added complexity of additions of large numbers.
2See Section 3.6 for a discussion on the accuracy of the result.
59
```

### PDF page 61

```text
can be extracted from a stock, given the number of cards in the stock and
the number of cards per flip. We derived a closed formula for cases where
the number of cards in the stock is more than once and at most twice
the number of cards per flip. We found an elegant recursion for the cases
where the number of cards in the stock is more than twice and at most
three times the number of cards per flip. For the general case we demon-
strate that a lower bound is given by the Pfaff-Fuss-Catalan sequences.
Tight bounds on the number of permutations that can be extracted from
a Klondike Solitaire stock for larger numbers of cards are still to be estab-
lished.
We discovered a class of games that can trivially be completed by per-
forming only moves directly to the foundation, when playing with the
rule that every card in the stock is accessible at all times (coined no-braider
games). Looking into smaller cases, we found that the relation is symmetri-
cal and we discovered two fascinating patterns. The first pattern concerns
situations in which all starting piles and all ending piles are of height 2, for
which we found both a recursive definition and a closed formula, but we
have not yet formally proved equality. The second pattern regards situa-
tions in which we have two starting piles and two ending piles, all of equal
height. We found that the number of possible configurations appears to be
either a square or twice a square, depending on whether the height is even
or odd respectively. We cannot explain the second pattern. Also, we would
be interested in an elegant interpretation of the closed formula for the first
pattern.
We were not yet able to compute the number of no-braider games for
the version of Klondike Solitaire where the piles on the tableau are of
height 1 through 7 and the stock contains 24 cars which are flipped per
three, due to time constraints, but we believe that given our current in-
sights and understanding, it will be only a matter of time before we will
be able to do so.
60
```

### PDF page 62

```text
Bibliography
[1] S.S. Skiena, The Algorithm Design Manual, Springer, Corrected ed.,
1997.
[2] R. Bellman, Dynamic Programming, Princeton University Press,
Princeton, New Jersey, 1957.
[3] H.S. Wilf, Generatingfunctionology, Academic Press, 1994.
[4] J. Levin, Solitaire-y confinement, 2008,
http://www.slate.com/id/2191295, retrieved Jun. 2012.
[5] Microsoft, Earnings Release FY12 Q2, 2012-01-19.
[6] X. Yan, P. Diaconis, P. Rusmevichientong, B. Van Roy, Solitaire: Man
Versus Machine, Solitaire: Man Versus Machine, Advances in Neural
Information Processing Systems 17, MIT Press, pp. 1553–1560, 2005.
[7] R. Bjarnarson, P. Tadepalli, A. Fern, Searching Solitaire in Real Time,
International Computer Games Association Journal 30(3), pp. 131–
142, 2007.
[8] Oberon Games and Microsoft Corporation, Microsoft Windows, Ver-
sion 6.1 (Build 7600).
[9] U. Latif, The probability of unplayable solitaire (Klondike) games,
2004, http://www.techuser.net/klondikeprob.html,
retrieved Dec. 2010.
[10] P.B. Donkersteeg, Klondike strategies using Monte Carlo techniques,
Master Thesis, 2010, retrieved Dec. 2010.
[11] C. Aguilar, Vectorized Playing Cards 1.3, Licensed under LGPL 3,
http://code.google.com/p/vectorized-playing-cards/,
retrieved Feb. 2012.
[12] Wikipedia – Inclusion-exclusion principle,
http://en.wikipedia.org/wiki/Inclusion_exclusion,
retrieved Feb. 2012.
[13] Wikipedia – Dihedral group,
http://en.wikipedia.org/wiki/Dihedral_group,
retrieved Aug. 2012.
61
```

### PDF page 63

```text
[14] Wikipedia – Factorial number system,
http://en.wikipedia.org/wiki/Factorial_base,
retrieved Apr. 2012.
[15] Wolfram Alpha LLC. 2009. Wolfram—Alpha.
http://www.wolframalpha.com, retrieved May 2012.
[16] A. Fog, Pseudo random number generators,
http://www.agner.org/random/, retrieved Feb. 2012.
[17] P.M. Fenwick, A New Data Structure for Cumulative Frequency Ta-
bles, Software: Practice and Experience, Vol. 24, 1994, pp. 327–336.
[18] D.H. Greene, D.E. Knuth, Mathematics for the Analysis of Algo-
rithms, 2nd ed., Birkh¨ auser, 1982, p. 17.
[19] L.E. Sigler, (trans.) Fibonacci’s Liber Abaci. Springer-Verlag. Chapter
II.12, 2002, pp. 404–405.
[20] M. Livio, The Golden Ratio: The Story of Phi, the World’s Most As-
tonishing Number. New York, Broadway Books, 2002, p. 108.
[21] Wikipedia – Catalan numbers,
http://en.wikipedia.org/wiki/Catalan_numbers,
retrieved Jul. 2012.
[22] R.L. Graham, D.E. Knuth, O. Patashnik, Concrete Mathematics.
Addison-Wesley, Reading, MA, 1990, pp. 200, 347.
[23] R.P. Stanley, Enumerative Combinatorics, Vol. 2, Cambridge Stud-
ies in Advanced Mathematics 62, Cambridge University Press, Cam-
bridge, 1999, p. 212.
[24] Wikipedia – Memoization,
http://en.wikipedia.org/wiki/Memoization,
retrieved May 2012.
[25] The Online Encyclopedia of Integer Sequences,
http://oeis.org/, retrieved Jul. 2012.
[26] Regular Graph, Wolfram MathWorld,
http://mathworld.wolfram.com/RegularGraph.html,
retrieved Jun. 2012.
62
```
