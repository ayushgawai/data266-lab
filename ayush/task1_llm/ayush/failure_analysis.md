# Task 1 failure cases

Three samples from the 20 generations in `outputs/generations.txt`. Temperature 0.8, prompts such as "Once upon a time" and "A girl named".

## 1. Names and pronouns flip

Sample 1 opens as Daisy, a girl, then switches to "he" with no new character.

> there was a princess named Daisy. Daisy loved to explore the colors and ran over to do it. He would stay in his way until he finally wanted to help her.

Sample 5 does the same with the name itself:

> A girl named Lucy. She played with her toy car and had a good named Rax. Max loved to play outside

The local statistics are TinyStories-like, but the model does not keep the entity it just introduced.

## 2. The scene cannot be both places

Sample 20 puts the character in the park and in her room in one sentence.

> she was walking in the park when she saw dogs in her room.

Sample 3 does the same with a tree inside the bedroom:

> she went to her room and saw a big tree and wanted to go to the park.

The next-character loss is low, so these still look like fluent kid-story sentences. The constraint that is missing is consistency across more than a few words.

## 3. Spelling and loops

Sample 18 invents words and then glues "A" onto "little":

> The zittle girl said "Thank you, you're so smart, Mommy."
> From that day on, Alittle girl loved to play outside together.

Sample 10 keeps saying "the little girl" until the sentence stops meaning anything:

> I'm sorry, the little girl and we want to enjoy the little girl and the little girl wore able to find our own shop.

A couple of samples also emit the byte sequence `â€œ` where a quotation mark should be. Those characters are in the 108-character vocab, so the model is copying a bad quote encoding from the source text rather than closing quotes on purpose.
