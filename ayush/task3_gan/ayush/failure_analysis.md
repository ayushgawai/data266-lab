# Task 3 failure analysis

Two failure modes from the submitted epoch-100 samples and the cycle numbers.

## The photo is still a painting

The Houses of Parliament Monet comes back as a photograph with a red sky, and the towers stay in the right place. The water and the sky still break into round color blobs. That is leftover brush texture. It is why Monet-to-photo FID is still high (local A2B ~85 on the submitted run) and not a small number. The discriminator is a patch network, so it can accept a locally photo-like patch while the global texture stays painted.

## The Monet painting changes the scene

The landscape photo keeps the horizon and the trees, then the generator lays a pink woven texture over the field and softens the buildings. Final-epoch cycle L1 on the submitted run is about 0.076 (identity about 0.052), so a round trip still does not return the same pixels. Identity loss limits the color shift, but it does not force the layout to stay exact. A higher cycle weight would pull the reconstruction down and would also make the translations look more like copies of the input.
