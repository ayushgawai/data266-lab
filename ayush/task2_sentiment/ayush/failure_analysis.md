# Task 2 failure analysis

Twenty mistakes from the attention model, five in each required bucket. Lines are in `outputs/bilstm_attn_errors.jsonl`.

## Confident false positives

Gold is negative and the predicted positive probability is above 0.998. The attended words are praise words.

> i had come here once before when they first opened and they were absolutely delicious ... and they no longer exist

Attention sits on delicious and absolutely. The review is about a closed restaurant.

> good poker room and great structured tournaments some of the wait staff is a little ride they carded my friend for a bottle of water

great and good outrank carded. The complaint is the staff.

Error type: a few positive unigrams outweigh the actual point of the review. A testable fix is to up-weight contrast and negation spans, or to train with a longer context instead of truncating at 256.

## Confident false negatives

Gold is positive and the predicted probability is under 0.002.

> the margaritas wont do ya dirty

wont and dirty are the top tokens. This is slang praise.

> their vacuums can really suck thank goodness

suck and ruined dominate. "suck" here means the vacuum works.

> its a nice buffet ... but variety was just ok meh

meh and ok beat nice.

Error type: words that are negative in isolation. The same fix as above, plus keeping multiword praise ("not bad", "can really suck") as a phrase instead of separate tokens.

## Near threshold

All five sit between 0.498 and 0.499 and are gold positive predicted negative. They are 4-star or "good but" reviews:

> it was ok but the bread is what made me give it four stars

> the food was decent but a little bit bland ... i would probably give them 4 stars

Error type: the label is positive while the text is mixed, and the model lands on the boundary. These are not a training failure in the same way as the 0.999 mistakes. A calibration pass would move the threshold, not the features.

## Long-review failures

All five are over 200 tokens, gold positive, predicted negative, probability under 0.01. The top weights are the early complaint words: bad, overpriced, insulted, disappointing, fraudulent, negligent.

> original post also see update below new south is a go to dinner place ... however tonight was disappointing

The model reads the complaint and never reaches the part that is still a 4-star review. This matches the slice table: long-review macro F1 is 0.921 against 0.941 on short reviews. Truncation at 256 tokens is the concrete cause when the verdict is at the end.
