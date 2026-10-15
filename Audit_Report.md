# Audit: report, notebooks and rubric

Checked: the report, the main notebook (124 cells), the error-analysis notebook (25 cells), and the brief and rubric.

## A. Fixed in the new report file (Yoruba_Sentiment_Report.md)
1. Section 4.2 "seen" definition was incomplete. It now lists the three checks, including cosine similarity of at least 0.90.
2. AfroXLMR was reported as 0.690. The notebook table gives 0.696 (run B: 5e-5, 8 epochs). Run A (3e-5, 5 epochs) scored 0.685. Both are now reported.
3. AfriBERTa seed standard deviations were 0.004 and 0.003. The notebook gives 0.005 (original text) and 0.004 (diacritics removed).
4. AfriBERTa gain on tweets without tone marks was 0.046. The notebook gives 0.725 against 0.677, which is 0.048.
5. Routing code-mixed tweets to AfroXLMR was reported as +0.004 to +0.006. The output is -0.002 to -0.001 on validation and +0.007 to +0.009 on test.
6. "AfroXLMR is better on code-mixed tweets" holds on test for both runs, but on validation only run A (3e-5) is ahead. Run B scored 0.712 there. The text now says it is not stable.
7. Added per-class precision, recall and F1 for the deployed model (rubric asks for these metrics).
8. Added a project-links line, a "Resources used and own work" subsection and library, model and dataset references.

## B. Problems in the main notebook that a grader or viva could hit
1. Several markdown "What this tells us" cells quote numbers that differ from the code output directly above them. They look like they come from an earlier run. Examples:
   - Cell 70: AfriBERTa validation 0.721 (output 0.717).
   - Cell 73: AfroXLMR test 0.688 (output 0.685).
   - Cell 76: run B validation 0.698 and test 0.690 (output 0.684 and 0.696).
   - Cell 79: validation std 0.003 (output 0.002).
   - Cell 88: no-tone means 0.678 and 0.724 (output 0.677 and 0.725).
   - Cell 106: routing gains of +0.004 to +0.006 (output -0.002 to +0.009).
   - Cell 115: AfroXLMR 0.690 (output 0.696).
   Cell 115's summary should be corrected first, since the report followed it. I did not edit your cells.
2. Cell 74 says the better AfroXLMR run on validation is carried forward. That is run A (0.692), but run B (0.684) is used in the final table and the routing. The report now states this openly.
3. Cell 18 still says "the test split is close to clean". Cell 43 shows it is a stripped copy of the raw text. Add a one-line pointer to cell 43.
4. Cells 46, 49, 100, 116 and 118 speak of "my friends will check" and the readers' answers as if they were coming. The audit was not completed. Add a one-line note that it was not completed.
5. The 8.2 markdown (cell 94) mixes numbers (0.631 and 0.598 against output 0.631 and 0.600). It is minor.

## C. Error-analysis notebook
1. `import base64` and `HTML` are imported inside cell 23. Move them into the imports cell.
2. The checkpoint path is hardcoded for Kaggle. Add a comment that it must be changed.
3. Cell 21 says "my reading of the Yoruba". Use neutral wording, since the glosses were not checked by a reader.
4. Cell 1.1 has no header markdown cell, and the last cell is empty.

## D. Rubric check (what is still open)
| Rubric item | Status |
|---|---|
| Problem, language and dataset (8) | Strong. Add explicit target users in the Introduction (one sentence). |
| Model development (10) | Strong. Hyperparameters are in the notebook; report gives the recipe. |
| Baseline and experimentation (8) | Strong: majority, word LR, tuned LR, two transformers, 3 problem experiments. |
| Evaluation and error analysis (7) | Strong. Qualitative examples of correct predictions are still thin. Add 3 to 4 examples if you have space. |
| Research report (10) | Needs the links filled in (GitHub, Space, video). |
| Web deployment (7) | Space works. Check the final link opens in a private window. |
| Code and GitHub (5) | Needs a top-level requirements.txt, README with links and the notebooks uploaded. |
| Technical understanding (5) | Viva: be ready to explain the seen/unseen split, why diacritics removal, why LR ties with transformers, why macro-F1. |

## E. Brief inconsistency
The brief gives the demo length as 7 to 10 minutes in one place and 10 to 15 in another. Aim for exactly 10 minutes, which satisfies both.

## F. Not verified
- Whether the app edit (agreement note and AfriBERTa box title) was applied to your Space. Section 8.1 of the report describes it. Check the live app.
- The Pang et al. (2002) and Das and Chen (2001) negation attribution rests on what I know of those papers; I could not read the full text here. It is the standard citation for the technique.
