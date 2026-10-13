# Public data attribution

The source observations are from **Dingdong-Inc, FreshRetailNet-50K**, provided under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- Dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Fixed source revision: `08c1fab7f9257bc73679d415d65d644165d351d4`
- The download script verifies the original train and eval files against SHA-256 hashes recorded in the frozen protocol.

This repository includes a filtered cohort, engineered features and result tables derived from those public observations. We select 312 store–SKU series, construct historical features, and generate predictions; these are modifications and are not an endorsement by the original provider. The raw source files are downloaded separately and are excluded from Git and the Docker build context.

The source-data license applies to redistributed source data and its adapted datasets. It does not imply that the original dataset authors supplied or endorsed this project's code, models, or conclusions.
