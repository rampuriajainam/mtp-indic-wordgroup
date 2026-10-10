# Annotation packet sources

FLORES rows are subsets of the original FLORES-200 devtest release, downloaded from the public archive linked by [Meta's original repository](https://github.com/facebookresearch/flores/blob/main/flores200/README.md). Attribution: NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine Translation* (2022). Original FLORES-200 license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The packet preserves original sentence text after trimming surrounding whitespace; the added rule labels are unreviewed research suggestions. `source_id` is the zero-based row within the original language file.

IndicCorp rows come from [`ai4bharat/IndicCorpV2`](https://huggingface.co/datasets/ai4bharat/IndicCorpV2), config `indiccorp_v2`, language split `hin_Deva`/`mar_Deva`, raw rows 1,000–5,000 inclusive. Attribution: AI4Bharat / IndicCorpV2 dataset authors. `source_id` is the raw stream row; corpus separators are skipped after applying the fixed row window.

IDs and source provenance are retained in the main packets. Om's blind packet intentionally contains only ID and text; it maps back to the Hindi main packet without exposing labels. Sampling is seeded and duplicate-free. Gold review, annotation and agreement have not yet been performed.
