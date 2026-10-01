# Lab 1 resources map

Updated 2026-09-15. Ayush work only. Sneha owns her member folders.

## Local layout

```
Lab1/
├── DATA266_Lab1_Fall_2026 (1).pdf
├── DATA266_Lab1_Plan_Ayush (1).html|.txt
├── DATA266_Lab1_Plan_Sneha (1).html|.txt
├── RESOURCES_STATUS.md
├── repo/                            ← shared team repo scaffold (ayush folders)
└── resources/
    ├── papers/
    ├── datasets_info/
    ├── web/
    └── kaggle/
        ├── pages/                   ← overview/rules/data mhtml + extracts
        ├── download/
        └── extract/                 ← monet_jpg, photo_jpg, real_stats.npz
```

## Decisions locked
- Task 2 dataset: **Yelp polarity**
- Task 1 split: **100K train / 10K val stories (rows)**. Each HF row is one story.
- Kaggle: self-report FID+MiFID in 1-row CSV. Up to 5 submits/day per team. Rank by score (lower better).

## Still open
- Exact CSV column names for Kaggle (likely `fid,mifid`)
- Team GitHub remote + team number
- GPU lab booking
- Staff did not ship `eval_cyclegan_fid.py`. We write our own using `real_stats.npz`.
