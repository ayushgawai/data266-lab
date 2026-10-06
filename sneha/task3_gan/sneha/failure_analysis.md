# Task 3 — Shortcomings, with image evidence (Sneha)

Generators: `checkpoints/generators_best.pt` from Colab step A (Monet → photo from epoch 30, photo → Monet from epoch 50). Images are the
submitted outputs in `outputs/pred_A2B/` and `outputs/pred_B2A/`, named by their input's filename stem.

To find examples without choosing them by eye, I measured how much each output changes its input (mean absolute pixel difference over
RGB, 0–255) for all 300 images per direction and looked at both ends of that ranking, plus the first images by filename
(`outputs/translations_first4.png`):

| Direction | Median change | 10th–90th percentile | Least changed | Most changed |
|---|---|---|---|---|
| Photo → Monet (B2A) | 5.2 | 1.9–11.6 | `126fb64cb8` 0.8, `70d01a9f18` 1.0, `e2be127e28` 1.0 | `d5f266c340` 14.5, `58ae62fe43` 15.1, `41da9742ed` 16.0 |
| Monet → photo (A2B) | 12.6 | 4.5–19.1 | `910610e827` 1.7, `9ae6552353` 2.0, `d5b0c260a0` 2.1 | `1f22663e72` 23.5, `4ad8b366c1` 23.9, `252d9a4abc` 26.1 |

## Failure mode 1 — type: weak stylisation (near-identity mapping)

**Images:** `pred_B2A/126fb64cb8.jpg`, `pred_B2A/70d01a9f18.jpg`, `pred_A2B/910610e827.jpg`; and the photo → Monet row of `translations_first4.png`.

**Observation:** the outputs are nearly copies of their inputs. `126fb64cb8` (a long-exposure seascape) changes by 0.8 / 255 per pixel and
`910610e827` (Monet's snowy houses) by 1.7: no brushwork is added to the photo and none is removed from the painting. It is the typical
case, not an outlier — half of all photo → Monet outputs change by less than 5.2 / 255. The metrics agree: content cosine 0.938 and LPIPS
0.111 for photo → Monet mean the output is perceptually very close to the input, and density 0.283 / coverage 0.420 mean the outputs sit
outside the real Monet distribution, which is why photo → Monet FID is the worse direction (122.38).
I think three things combine: the U-Net's skip connections let the generator pass the input straight to the output; the identity term
(weight 5) rewards leaving images unchanged; and once the discriminators saturated (D losses near 0.02 from epoch 30) their gradient
no longer pushed the generator towards Monet texture, so the cheapest way to lower the total loss was to copy.
**Experiment:** retrain with the identity weight set to 0 (everything else equal) and compare the median pixel change and photo → Monet FID
at the same epoch; if the identity term drives the copying, both should move clearly.

## Failure mode 2 — type: texture artifacts (a repeated dotted pattern)

**Images:** `pred_B2A/41da9742ed.jpg` (night sky over a bridge), `pred_B2A/0b1669c3ea.jpg` (sky above a snowy ridge).

**Observation:** smooth, dark or flat regions get a regular grid of small green, blue and pink star-shaped dots that is not in the input.
In `41da9742ed` it covers the whole black sky, which is why this image is among the most changed (16.0 / 255), even though the bridge
itself is barely altered; in `0b1669c3ea` it sits in the sunset sky. The pattern repeats at a fixed spacing whatever the image content,
so it comes from the network rather than the data.
Two causes are likely. The decoder upsamples with stride-2 transposed convolutions (`ConvTranspose2d(…, 4, 2, 1)`), which are known to
produce periodic, checkerboard-like patterns. And a generator that copies its input (mode 1) still has to lower its adversarial loss
somehow; a faint fixed high-frequency pattern is a cheap signal for a patch discriminator, which may also be where CycleGAN generators hide
information they need for reconstruction. FID and LPIPS penalise it only slightly, because it is low in amplitude.
**Experiment:** replace the transposed convolutions with nearest-neighbour upsampling followed by a 3×3 convolution, retrain for the same
epochs, and compare the frequency spectrum of outputs on flat skies: if the transposed convolutions cause it, the peaks at the pattern's
spacing should disappear.

## Failure mode 3 — type: content loss in fine detail (smearing and over-saturation)

**Images:** `pred_A2B/252d9a4abc.jpg` (Venice, the most changed Monet → photo output, 26.1 / 255), `pred_A2B/1f22663e72.jpg`, `pred_A2B/4ad8b366c1.jpg`.

**Observation:** where Monet → photo does change the image, it does so unevenly. In `252d9a4abc` the water and sky become smoother and
more saturated, and contrast goes up, which is a step towards a photo; but thin structures such as the mooring poles and the building
edges are smeared into streaks, and the result looks neither like a photo nor like the painting. This direction has the higher LPIPS
(0.132 vs 0.111) and lower content cosine (0.895 vs 0.938), so it is where content is lost. Its FID is still the better one (107.90),
because removing brushwork moves the image towards photo statistics even when the detail suffers.
I think the generator learned a global "smooth and saturate" operation rather than a content-aware one: with only 300 Monets and a
saturated discriminator, nothing teaches it how real photos render fine structure.
**Experiment:** translate the same three images with the epoch-120 generator (`checkpoints/generators_final.pt`, after the full learning-rate
decay); if the smearing comes from training at a high constant learning rate, the decayed generator should keep thin structures while
still smoothing the brushwork.
