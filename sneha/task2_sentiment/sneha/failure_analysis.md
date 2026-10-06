# Task 2 — 20-error review (Sneha)

The 20 candidates are copied from `outputs/error_review_candidates.md`, written by the notebook (section 14) by fixed rules: 5 confident
false positives, 5 confident false negatives, 5 near-threshold errors and 5 from the worst slice. Full review texts are in that file.

Model reviewed (best validation macro-F1): **bigru** (0.9377) · checkpoint: `checkpoints/bigru_best.pt` · worst slice: **short** (test macro-F1 0.9308)
Near-threshold errors inside [0.45, 0.55]: **235** (all 5 below come from the band). Test rows index the official `test.csv` (0-based); the brackets give the true label and the slices.

Suggested error types: sarcasm / irony · negation · mixed sentiment (contrast) · label noise ·
sentiment carried by a rare or truncated word · review cut off at max_len · domain-specific phrase.

## Confident false positives

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 1 | 29330 (negative; short) | 0.9998 | Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012 | label noise | The text is wholly positive ("love the place", "worth a try"); no model can recover a 1–2-star label from it. Test: hand-label the 100 most confident errors and estimate the label-noise rate; exclude those rows and re-score. |
| 2 | 408 (negative; negation, contrast) | 0.9995 | Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago. Copper is definitely still my place, … | mixed sentiment (contrast), comparison with a competitor | Praise ("pretty good", "amazing") outweighs hedges ("fine enough", "so-so") word by word; the negative verdict is relative to another restaurant. Test: attention pooling instead of max-pooling, scored on the contrast slice. |
| 3 | 12480 (negative; short, contrast) | 0.9993 | my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome! | label noise | Positive throughout ("good", "great", "awesome") apart from "a little on the small side". Same test as #1. |
| 4 | 8426 (negative; long, negation, contrast) | 0.9987 | Saturday / Sunday AYCE brunch In true las vegas fashion, you get a flat rate to eat your heart out. For Strip food, main menu items seem reasonably priced and the brunch is cheap at $29.99. There are a few different … | mixed sentiment, verdict at the end | A long, mostly positive description; the disappointment ("would think … more impressive") comes last. Test: score the last 256 tokens instead of the first 256 and compare on long reviews. |
| 5 | 3247 (negative; short) | 0.9987 | This is a great place to get your dog's groomed if you can get them in, the wait for an appointment is ridiculous. It's almost worth the extra price to get them in sooner. :( | mixed sentiment; cue removed by cleaning | "great place" dominates; the complaint ("ridiculous") and the closing ":(" carry the label, and cleaning deletes ":(". Test: keep emoticons as tokens (":(" → `<sad>`, ":)" → `<smile>`) and re-score. |

## Confident false negatives

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 6 | 33573 (positive; short) | 0.0002 | For being a DUMP, should expect much more. Flys, stink, garbage, dirt, and everything that comes with. Salt River... Keepin it real dumpy! | sarcasm / irony | Every word is negative ("DUMP", "stink", "garbage"); the praise is ironic ("Keepin it real dumpy!"). Word-level models can't see it. Test: check whether charcnn and Ayush's attention model also miss it; if all do, it needs context beyond word cues. |
| 7 | 22807 (positive; negation, contrast) | 0.0002 | EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server … | label noise (stale rating) | "EDIT: … Horrible service": the text was rewritten after a positive rating. Test: flag reviews containing "EDIT"/"UPDATE" and measure their error rate against the rest. |
| 8 | 20383 (positive; negation, contrast) | 0.0005 | After some nightmares with Chase, who inexplicably closed the checking account of the non-profit organization for students in which I am involved, I decided to try to take the account to Wells Fargo. This branch is … | sentiment about another entity | The negative words ("nightmares", "awful") describe Chase; the praise for this branch is milder and at the end ("very easy", "Much appreciated!"). Test: attention pooling, checked on reviews that name a competitor. |
| 9 | 11401 (positive; negation, contrast) | 0.0005 | Perhaps my expectations were too high because of all the hype I'd heard, but I have to say I was a little disappointed. We did a girls' night out and were so excited about this new great wine/chocolate combo thing, but … | 3-star boundary (mixed sentiment) | Mostly complaints ("disappointed", "waited forever", "cold food") with some praise; in this dataset 3 stars count as positive, so a lukewarm review is "positive". Test: error rate on reviews with both strong positive and strong negative words, against the rest. |
| 10 | 7450 (positive; negation) | 0.0005 | Rio has very slow elevator, when it's busy you may need to wait for a long time. we stayed at the honeymoon suite. it has two living room and 1 dining room. the Jacuzzi bath tub is almost in the room because there no … | verdict at the end, domain-specific phrase | Opens with a complaint ("very slow elevator"); the positive verdict is a phrase, not a sentiment word ("a room I want to come back for"), and "back" is a stopword. Test: drop "back" and "again" from the stopword list and re-score. |

## Near-threshold errors

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 11 | 34614 (positive; negation) | 0.4996 | Just so you know...you will wait. I waited 45 minutes for food for five people. Just so you REALLY UNDERSTAND.....IT IS WORTH IT. I tasted the ginger, scallions, the garlic. no excess oil in the noodles or fried rice. … | negation, emphasis lost | "You will not be disappointed" and the shouted "IT IS WORTH IT" decide it; "disappointed" stays negative under bag-of-words, and lowercasing removes the emphasis. Test: mark negation scope (prefix `NOT_` to words after "not" until punctuation) and re-score the negation slice. |
| 12 | 4358 (negative; negation, contrast) | 0.5004 | I have stayed here many times, but I will now find somewhere else to stay. The front desk people are never very friendly. Our first room of this last visit was nasty. The bathroom tile was coming up and the door hinges … | mixed sentiment, negation in a contraction | A long list of complaints and some praise; the verdict "I won't be back" becomes "wont" (apostrophe removed) and "back" is a stopword, so the verdict is lost. Test: expand contractions ("won't" → "will not") before cleaning, as Ayush does. |
| 13 | 5890 (positive; short) | 0.4993 | OK: I like the Ren Festival. I still have a T shirt from 1991 which is in very good condition. My one quibble is that it get so damn crowded; oh well. I still go at least once a year. :) | mixed sentiment, cue removed by cleaning | Mild praise ("I like", "still go") against "so damn crowded"; the closing ":)" is deleted and "still" is a stopword. Test: keep emoticons (as in #5) and "still". |
| 14 | 16495 (positive; long, negation, contrast) | 0.4991 | I've been to Stax several times in the past few months, and every time I've been incredibly pleased with my experience -- the food has been excellent, the service has been great, and the bill has come out to be very … | mixed sentiment (contrast) | Praise, then a long complaint ("however", "SO LOUD", "disappeared"), then "Still … remains a favorite" — and "still" is a stopword. Test: attention pooling on the contrast slice; remove "still" from the stopword list. |
| 15 | 27066 (positive; negation, contrast) | 0.4984 | The Smart Stork is yet another cute children's clothing boutique in the Bruntsfield area. If you're looking for gifts for newborns then this is the kind of place you could get gorgeous presents or christening items that … | negation in a contraction, domain phrase | "wouldn't begrudge" (a double negative = positive) becomes "wouldnt begrudg"; "terrible tot" uses "terrible" affectionately; "not the cheapest" reads as negative. Test: expand contractions and mark negation scope (#11, #12). |

## Worst-slice failures (slice: short)

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 16 | 3223 (positive; short, contrast) | 0.0020 | I've been here two times. The first time was really good, but the second was very boring flavored. I had some lemon chicken that was mediocre. For the price, the last time turned me off, but will probably go back again. | 3-star boundary (mixed sentiment) | "mediocre", "boring", "turned me off", then "will probably go back again" — both "back" and "again" are stopwords, so the positive ending disappears. Test: remove "back"/"again" from the stopword list. |
| 17 | 22269 (negative; short) | 0.9977 | Second trip to the bar v ased on a promo of free play. Gambling was good, great Pandora music, and plenty of tv. Bar keeps are to much chatty kathys among themselves. | label noise or mild complaint | Mostly positive ("good", "great", "plenty"); the one complaint ("too much chatty") is mild. Same test as #1. |
| 18 | 22707 (positive; short, contrast) | 0.0025 | Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh.. | 3-star boundary, slang | "nice", "just ok", "Meh..": lukewarm, labelled positive. "Meh" is rare in training. Test: error rate of reviews containing "meh", "ok" or "okay" against the rest. |
| 19 | 7126 (positive; short, contrast) | 0.0026 | This is a very well run airport, but the security line can get way too long. The food choices in the terminals is just ok. | 3-star boundary (mixed sentiment) | "very well run" against "way too long" and "just ok"; the negatives come last. Test: as #9. |
| 20 | 35290 (negative; short, negation, contrast) | 0.9972 | Stopped by and was surprised at the size of this store. The technology is pretty neat too. BUT I was very surprised at the prices I looked at. I found everything to be priced high, the staff did not mention the bidding … | mixed sentiment (contrast) | Mild praise ("pretty neat") then "BUT … priced high"; the capitals that mark the turn are lowercased. Test: weight the clause after "but" more (contrast-aware pooling) and re-score the contrast slice. |

## Patterns across the 20

- **Some errors are the labels, not the model (#1, #3, #7, #17).** These reviews read as clearly positive or clearly negative, opposite to
  their stars. No text model can fix them, and they set a floor under the error rate.
- **The 3-star boundary (#9, #16, #18, #19).** In this dataset 3 stars count as positive, so lukewarm or mixed reviews are labelled
  positive. Most short-slice errors (16–19) are of this kind: with 50 words or fewer, one strong word decides the prediction.
- **Mixed reviews where the verdict comes after "but" or at the end (#2, #4, #8, #10, #14, #20).** BiGRU max-pooling keeps the strongest
  feature over all positions, so it can't tell the closing verdict from a complaint in the middle.
- **My cleaning removes cues (#5, #11, #12, #13, #15, #16).** Emoticons are deleted, capitals are lowercased, "won't" and "wouldn't" lose their
  negation as "wont" and "wouldnt", and scikit-learn's stopword list drops words I did not protect in `keep_words`: back, again, still,
  well, enough, only, even. This is also the most likely reason the ablation without stopword removal and stemming scored higher (validation
  macro-F1 0.9269 vs 0.9212).
- **Sarcasm (#6)** is the only error that needs more than word cues.

The fixes that would test most at once are preprocessing changes: expand contractions before removing apostrophes, keep emoticons as
tokens, and take back, again and still out of the stopword list; then re-run bigru and compare test macro-F1 and the slice error rates.
