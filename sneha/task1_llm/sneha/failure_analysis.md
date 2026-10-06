# Task 1 — Failure analysis (Sneha)

Checkpoint: `checkpoints/gpt_best.pt` (epoch 30) · decoding: temperature 0.8 for samples 01–20, greedy for greedy 01–05
(see the header of `outputs/generations.txt`). The three snippets below are pasted verbatim from that file.

## Failure 1 — type: repetition

**Sample:** `greedy 04`, prompt `The sun was shining and`

```
The sun was shining and said, "I want to play with you, but I want to play with you." The bunny was so happy to see the bunny and the bunny was so happy to see the bunny and the bunny was so happy to have a new friend.
```

**Observation:** with greedy decoding the model falls into loops: "I want to play with you" twice in one sentence, then "the bunny
was so happy to see the bunny" twice. All five greedy samples loop, and four of them (01, 03, 04, 05) produce the same stock passage, word for word
("there was a little girl named Lily. She loved to play with her toys and play with her toys"). Greedy decoding
always takes the single most likely next character; once a phrase has appeared, the copy of it inside the 128-character context makes
repeating it even more likely, so the loop reinforces itself. The temperature-0.8 samples rarely do this (repeated 4-gram rate 0.0127
over the 20 samples), which points at the decoding, not the training.
**Experiment:** generate the same 5 prompts and seeds greedy, at temperature 0.8, and with top-p 0.9, and compare the repeated 4-gram rate;
if decoding is the cause, the greedy rate should be far higher with the same checkpoint.

## Failure 2 — type: loss of coherence

**Sample:** `sample 01`, prompt `Once upon a time`

```
Once upon a time, in a small house, there was a graceful cat. The bird loved to run and said, "Good job, Tim. I made a loud noise." Tim looked around the cat and said, "My name is Sam. The cat is a great idea." He said, "Thank you, dog."
Tim and Sam felt sad. They played with the ball all day long. They wanted to play with it. They saw the ball and wanted to see the ball.
The ball was very sad and said, "Let's play with it all the music." The ball said, "I want to share the ball too!" The ball was so happy.
```

**Observation:** each sentence is locally fluent, but the story doesn't hold together. The cat introduced in the first sentence is
replaced by a bird, then by Tim, who says his name is Sam and thanks a dog that never appeared; by the third paragraph the ball has become a
character that talks. Nouns and names change from sentence to sentence. With a 128-character context the model only ever sees about the
last two sentences, so it can't track who the story is about: it continues whatever names are in view, and falls back on TinyStories'
most frequent ones (Tim, Sam, Lily, the ball) when the earlier ones have scrolled out.
**Experiment:** train the same model with `block_size` 256 (everything else equal) and count, over the same 20 prompts and seeds, how many
distinct character names each story uses; if the context length is the cause, the count should drop.

## Failure 3 — type: broken grammar (and invented words)

**Sample:** `sample 10`, prompt `Lily wanted to`

```
Once upon a time, there was a pretty bear named Tim. Tim loved to eat long to his friends. One day, he went to the park with his friends, playing near the grill. Tim was very happy and he asked his mom to see him lost him to read the lock right.
<|endoftext|>
Once upon a time, there was a hosple cat named Fin was friendly. He loved
```

**Observation:** two kinds of error at once. Sentences break their grammar halfway ("Tim loved to eat long to his friends",
"he asked his mom to see him lost him to read the lock right", "a hosple cat named Fin was friendly"), and the model invents words that
don't exist: "hosple" here, "sheell" in sample 06 and "berriefl" in sample 08. A character model has to spell every word itself; with
top-1 accuracy 0.748, about one character in four is not the most likely one, and sampling at temperature 0.8 sometimes picks a low-
probability character in the middle of a word, after which the model continues the misspelling. The grammar breaks the same way at word
level: each word is plausible after the last few, but nothing holds the clause together.
**Experiment:** generate the same 20 prompts and seeds at temperatures 0.5, 0.8 and 1.0 and count the fraction of generated words that never
appear in the training slice; if sampling is the cause, the non-word rate should rise with temperature.
