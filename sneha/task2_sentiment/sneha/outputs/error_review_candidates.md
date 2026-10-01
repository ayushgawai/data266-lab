# Task 2 error review candidates: bigru (best validation macro-F1 0.9377)

Checkpoint `task2_sentiment/sneha/checkpoints/bigru_best.pt`. Test rows index the official test.csv (0-based). Worst slice by test macro-F1: **short** (0.9308). Near-threshold errors inside [0.45, 0.55]: **235** (all 5 from the band).

## 1. confident false positive
- test row: 29330 | true: negative | p(positive) = 0.9998 | slices: short
- review: Wow love the place and everything is very clean and new!  Great place to come and relax worth a try!  Cheers,  Eric Van Nguyen Visited April 2012
- error type: _(you)_
- testable fix: _(you)_

## 2. confident false positive
- test row: 408 | true: negative | p(positive) = 0.9995 | slices: negation, contrast
- review: Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago.   Copper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly, which is rare. Usually indian food, good indian good, takes at least 30-45 minutes. We got our order in like 30, which was great 'cause we were starving.   The Tikka Masala was spicy and pretty good, but it wasn't as thick and saucy as i like. The salad was just so-so. The Naan were all amazing. Definitely the highlight.
- error type: _(you)_
- testable fix: _(you)_

## 3. confident false positive
- test row: 12480 | true: negative | p(positive) = 0.9993 | slices: short, contrast
- review: my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!
- error type: _(you)_
- testable fix: _(you)_

## 4. confident false positive
- test row: 8426 | true: negative | p(positive) = 0.9987 | slices: long, negation, contrast
- review: Saturday / Sunday AYCE brunch  In true las vegas fashion, you get a flat rate to eat your heart out.   For Strip food, main menu items seem reasonably priced and the brunch is cheap at $29.99. There are a few different flavors to choose from for the All-You-Can-Drink Bottomless mimosas. They're $5 per person; but on multiple occasions, the $5 was comped for me and my dining partner. Maybe its a local discount?   At the beginning, you're served a plate of sliced fruit w/ a mango sauce and a plate of guava empanadas. Those empanadas are probably the best part of the menu! Not too sweet with the perfect crust and just a hint of cream cheese in the filling. After that I was a little under-enthused. The savory dishes start to taste the same after a while and the desserts are much too sweet. Their gimmick item is the Bacon Jalape\u00f1o PBJ. Peanut butter, jelly, egg, bacon, and a couple slices of mild jalapeno on a biscuit. Worth a try. It is AYCE after all.   You would think a Top Chef Masters restaurant would be more impressive. And given that they've won Best Brunch awards from Seven, LVRJ, & LVW, I expected this place to knock my socks off! While I can't judge the main menu food, I  ...
- error type: _(you)_
- testable fix: _(you)_

## 5. confident false positive
- test row: 3247 | true: negative | p(positive) = 0.9987 | slices: short
- review: This is a great place to get your dog's groomed if you can get them in, the wait for an appointment is ridiculous. It's almost worth the extra price to get them in sooner. :(
- error type: _(you)_
- testable fix: _(you)_

## 6. confident false negative
- test row: 33573 | true: positive | p(positive) = 0.0002 | slices: short
- review: For being a DUMP, should expect much more.  Flys, stink, garbage, dirt, and everything that comes with.  Salt River... Keepin it real dumpy!
- error type: _(you)_
- testable fix: _(you)_

## 7. confident false negative
- test row: 22807 | true: positive | p(positive) = 0.0002 | slices: negation, contrast
- review: EDIT: They really did change the service up since I last posted this.  Horrible service.  Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'  Sorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.
- error type: _(you)_
- testable fix: _(you)_

## 8. confident false negative
- test row: 20383 | true: positive | p(positive) = 0.0005 | slices: negation, contrast
- review: After some nightmares with Chase, who inexplicably closed the checking account of the non-profit organization for students in which I am involved, I decided to try to take the account to Wells Fargo. This branch is closest to my school, and the banker I worked with made the process of opening up a new non-profit account very easy, and I did not have to file all kinds of awful paperwork that Chase would have made me do. I have not had good experiences at every branch (7th St, just past Camelback, I'm looking at you), but this one let me keep the organization running, financially speaking, with a minimum of red tape. Much appreciated!
- error type: _(you)_
- testable fix: _(you)_

## 9. confident false negative
- test row: 11401 | true: positive | p(positive) = 0.0005 | slices: negation, contrast
- review: Perhaps my expectations were too high because of all the hype I'd heard, but I have to say I was a little disappointed. We did a girls' night out and were so excited about this new great wine/chocolate combo thing, but after waiting forever for a table we waited forever for a waitress, who utimately messed up our order and brought us first the wrong food and then cold food. When they finally got the food right, it was actually pretty goos, but by that time were were kind of over it. As for dessert, we ended up going with these chocolate martinis that were pretty rockin. My suggestion is head there for drinks but not for food.
- error type: _(you)_
- testable fix: _(you)_

## 10. confident false negative
- test row: 7450 | true: positive | p(positive) = 0.0005 | slices: negation
- review: Rio has very slow elevator, when it's busy you may need to wait for a long time. we stayed at the honeymoon suite. it has two living room and 1 dining room. the Jacuzzi bath tub is almost in the room because there no wall between the bathroom and bedroom. separate room for toilet and shower. and one more half bathroom in the living room. this will be a room I want to come back for.
- error type: _(you)_
- testable fix: _(you)_

## 11. near threshold
- test row: 34614 | true: positive | p(positive) = 0.4996 | slices: negation
- review: Just so you know...you will wait.  I waited 45 minutes for food for five people. Just so you REALLY UNDERSTAND.....IT IS WORTH IT.  I tasted the ginger, scallions, the garlic.  no excess oil in the noodles or fried rice.  The food is prepared FRESH, FRESH, FRESH! Everything is prepared so beautifully, so carefully....people, GO!!!!!!!  You will not be disappointed.  I live over fifty miles away and I will and I mean this, I WILL BE BACK!
- error type: _(you)_
- testable fix: _(you)_

## 12. near threshold
- test row: 4358 | true: negative | p(positive) = 0.5004 | slices: negation, contrast
- review: I have stayed here many times, but I will now find somewhere else to stay.  The front desk people are never very friendly.    Our first room of this last visit was nasty.  The bathroom tile was coming up and the door hinges were rusty and the paint was coming off the wall near the tub.  The tub had grout coming off.  No continuing maintenance is going on.   Our A/C did not work and we had to be transferred at 11:30 pm to another room.  This means that the maid did not report it not working or that they did not fix it.  Our second room was almost as bad.  The bathroom was in the same sorry shape, but the room was nicer.    The Casino is always fun and great priced for being small and off the strip.  The pool can get a little crowded, but the pool bar and hooters restaurant overlooking the pool are fun.  I won't be back until some remodel is done.  Going to try the "New" Tropicana.
- error type: _(you)_
- testable fix: _(you)_

## 13. near threshold
- test row: 5890 | true: positive | p(positive) = 0.4993 | slices: short
- review: OK: I like the Ren Festival. I still have a T shirt from 1991 which is in very good condition. My one quibble is that it get so damn crowded; oh well. I still go at least once a year. :)
- error type: _(you)_
- testable fix: _(you)_

## 14. near threshold
- test row: 16495 | true: positive | p(positive) = 0.4991 | slices: long, negation, contrast
- review: I've been to Stax several times in the past few months, and every time I've been incredibly pleased with my experience -- the food has been excellent, the service has been great, and the bill has come out to be very reasonable considering everything.  It makes sense then that I chose to go here last Thursday with my sister and coworkers for my birthday lunch. I had been looking forward to treating myself to one of their delish Veggie burgers all week!  I was a teensy bit disappointed by two things that day, however. 1. The music in the place was up SO LOUD that I could barely hear anyone at the table, and even though we asked our waitress to turn down the volume, I think she barely touched the dial, because I didn't notice any improvement whatsoever. And 2. Our service that day was not anywhere near what it had been on my previous visits. Our server simply disappeared once we were seated, so we didn't get to order for a looooong time, and I don't think she ever checked on us mid-meal.  Still, despite my last experience there, Stax still remains a favorite, and I'll be a frequenter as long as the service goes back to the way it was before. I'm keeping them at five stars for now, but ...
- error type: _(you)_
- testable fix: _(you)_

## 15. near threshold
- test row: 27066 | true: positive | p(positive) = 0.4984 | slices: negation, contrast
- review: The Smart Stork is yet another cute children's clothing boutique in the Bruntsfield area. If you're looking for gifts for newborns then this is the kind of place you could get gorgeous presents or christening items that they can keep until they're older and think back at their younger years.  They stock all manner of toys, pictures, room decorations, teddies, christening mugs. Most of the toys you wouldn't give to a terrible tot to play with but keep them for decoration, perhaps on a shelf in their room.   You could spend ages in here "oohing" and "awwing" at all the cuteness on offer and although it's not the cheapest store, the prices reflect the quality and I wouldn't begrudge paying the prices for the beautiful items for sale here.
- error type: _(you)_
- testable fix: _(you)_

## 16. slice: short
- test row: 3223 | true: positive | p(positive) = 0.0020 | slices: short, contrast
- review: I've been here two times. The first time was really good, but the second was very boring flavored. I had some lemon chicken that was mediocre.    For the price,  the last time turned me off, but will probably go back again.
- error type: _(you)_
- testable fix: _(you)_

## 17. slice: short
- test row: 22269 | true: negative | p(positive) = 0.9977 | slices: short
- review: Second trip to the bar v ased on a promo of free play.   Gambling was good, great Pandora music, and plenty of tv. Bar keeps are to much chatty kathys among themselves.
- error type: _(you)_
- testable fix: _(you)_

## 18. slice: short
- test row: 22707 | true: positive | p(positive) = 0.0025 | slices: short, contrast
- review: Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh..
- error type: _(you)_
- testable fix: _(you)_

## 19. slice: short
- test row: 7126 | true: positive | p(positive) = 0.0026 | slices: short, contrast
- review: This is a very well run airport, but the security line can get way too long.  The food choices in the terminals is just ok.
- error type: _(you)_
- testable fix: _(you)_

## 20. slice: short
- test row: 35290 | true: negative | p(positive) = 0.9972 | slices: short, negation, contrast
- review: Stopped by and was surprised at the size of this store. The technology is pretty neat too. BUT  I was very surprised at the prices I looked at. I found everything to be priced high, the staff did not mention the bidding process.
- error type: _(you)_
- testable fix: _(you)_
