# Sentiment Classification for Yoruba Tweets

Divine Ebube Ifechukwude · 2026-10-14

Project links: GitHub repository: [to be added] · Live system (Hugging Face Space): [to be added] · Demo video: [to be added] · Models: https://huggingface.co/Thevyne/yoruba-sentiment-models

## 1. Abstract

A word plus character logistic regression reaches macro-F1 0.732 on Yoruba tweets the models have not seen in training, matching fine-tuned transformers within noise, and both are deployed in a public web app.

This study classifies Yoruba tweets as positive, neutral or negative using the Yoruba portion of AfriSenti. 12.8% of the test tweets also appear in training, which inflates the full-test macro-F1 to 0.759, so all conclusions use the 3,938 unseen test tweets. On these, the best model (word and character TF-IDF with logistic regression, on text without tone marks) scores 0.732, against 0.720 to 0.721 for fine-tuned AfriBERTa (mean over seeds) and 0.696 for AfroXLMR; the top three lie within the noise level of about 0.015. Three problems were tested. Removing tone marks and underdots costs nothing overall and gains about 0.044 to 0.048 macro-F1 on tweets that have none; negation raises the error rate from 23.8% to 30.7%, and a scope-tagging fix did not help; the penalty for code-mixed tweets did not replicate across splits. Error analysis shows that neutral is the most confused class, that the two best models share 66% of their errors, and that the logistic regression is well calibrated (expected calibration error 0.014) while AfriBERTa is strongly overconfident (0.210). Label quality could not be measured. The models are served by a Gradio app on Hugging Face Spaces.

## 2. Introduction

This study builds a three-class sentiment classifier (positive, neutral, negative) for Yoruba tweets, measures it honestly, and deploys it as a public web app.

Yoruba is a tonal language with over 50 million speakers (Jimoh et al., 2026), yet it has few labelled resources for language technology (Joshi et al., 2020). Sentiment analysis of social media text is a practical use: it lets researchers and organisations see how people feel about public issues in a language that mainstream tools handle poorly. AfriSenti (Muhammad et al., 2023) now provides labelled Yoruba tweets, which makes it possible to build and compare models.

Three features of Yoruba tweets make the task harder than it looks, and each is treated as an explicit problem in this study: how it is measured, how the model handles it, and how it is tested as an experiment.

- Missing tone marks. Tone marks and underdots are part of Yoruba spelling, but tweets use them inconsistently, so the same word appears in several written forms.
- Negation. A negated tweet can contain positive words and still be negative, or the reverse, which is hard for models that rely on words.
- Code-mixing. Some tweets mix Yoruba with English or Nigerian Pidgin, which breaks models trained on one language.

The main contributions are:

1. An evaluation that separates test tweets seen in training from unseen ones. The score on the full test set (0.759) overstates performance by about 0.03 compared with the unseen tweets (0.732), and the gap between seen and unseen tweets is about 0.20 to 0.24 for every learned model.
2. A comparison of linear models and fine-tuned transformers (AfriBERTa and AfroXLMR) with noise levels measured, which shows that the best linear model matches the best transformer within noise.
3. Tests of the three problems: removing all diacritics costs nothing overall and helps on tweets without tone marks, negation causes the clearest increase in errors, and the code-mixing effect did not replicate across splits.
4. An error analysis showing that the two best models share two thirds of their errors, that one of them is well calibrated and the other strongly overconfident, and where the confusions lie.
5. A deployed web app and public repository that apply the same text preparation as the training data.

Section 3 reviews related work, Sections 4 and 5 describe the data and methods, Section 6 gives the experiments, Section 7 analyses errors, Section 8 describes the deployment, and Section 9 states limitations.

## 3. Related Work

Yoruba sentiment analysis builds on three lines of work: labelled corpora for African languages, pretrained models adapted to them, and long-standing techniques for classifying sentiment and handling negation.

### 3.1 Resources for Yoruba

Joshi et al. (2020) sort the world's languages into six classes by the amount of labelled and unlabelled data available, from class 0 (largely ignored) to class 5 (rich in resources). Yoruba is placed in class 2, the Hopefuls, which have only a small set of labelled datasets, as listed in Adelani (2022, Table 2.3). Yoruba has over 50 million speakers (Jimoh et al., 2026), so the gap between speaker numbers and available data is large.

For sentiment, NaijaSenti (Muhammad et al., 2022) is a Twitter corpus for four Nigerian languages: Hausa, Igbo, Nigerian Pidgin and Yoruba. AfriSenti (Muhammad et al., 2023) extends this to 14 African languages with more than 110,000 tweets, each labelled by three native speakers, and was the basis of SemEval 2023 Task 12. The paper notes that Yoruba tone is rarely fully written, which makes annotation harder, and reports an inter-annotator agreement of 0.65 for Yoruba, so some label disagreement is to be expected. For a related task, Yor-Sarc (Jimoh et al., 2026) is a gold-standard Yoruba sarcasm dataset of 436 instances, each annotated by three native speakers.

### 3.2 Pretrained models for African languages

AfriBERTa (Ogueji et al., 2021) is a multilingual transformer pretrained from scratch on 0.94 GB of text in 11 African languages, in small (97 million parameters), base (111 million) and large (126 million) sizes, built to show that pretraining on little data can still help low-resource languages. AfroXLMR (Alabi et al., 2022) takes the opposite route: it adapts an existing multilingual model to 17 African languages and 3 widely spoken languages by further training on text in those languages, and prunes the vocabulary of unused scripts to halve the model size. This study fine-tunes the base sizes of both on Yoruba tweets and compares them with linear models.

### 3.3 Sentiment classification and negation

Pang et al. (2002) showed that standard learning methods (Naive Bayes, maximum entropy and support vector machines) classify the sentiment of movie reviews better than hand-made baselines, while the task remains harder than topic classification. Their feature setup included a way of handling negation that marks the words between a negation cue and the next punctuation mark, a technique described earlier by Das and Chen (2001). This study tests a version of that idea for Yoruba (Section 6.3). Sarcasm is a further source of difficulty. Contextual models such as C-Net (Jena et al., 2020) use the surrounding context to detect it, and Yor-Sarc provides Yoruba data for the same purpose; sarcasm is outside the scope of this study.

### 3.4 Position of this study

The studies above provide the data and the models. What this study adds is an evaluation that separates tweets seen in training from unseen ones, which changes the headline score by about 0.2 macro-F1, together with direct tests of three problems specific to Yoruba tweets (missing tone marks, negation and code-mixing) and an error analysis of two deployed models.

## 4. Dataset and Data Preparation

The study uses the Yoruba portion of AfriSenti, a corpus of tweets labelled positive, neutral or negative, and its most important property is that many test tweets also appear in the training data, so scores on the full test set overstate what the models can do.

### 4.1 The data

The data come from the  configuration of  (licence CC BY 4.0), loaded from its Parquet export. Each record holds a tweet and a three-class label (positive 0, neutral 1, negative 2). AfriSenti covers 14 African languages with more than 110,000 tweets, each labelled by three native speakers with the final label taken by majority vote, and it was used for SemEval 2023 Task 12 (Muhammad et al., 2023). For Yoruba, the paper reports 15,130 tweets, an inter-annotator agreement (free-marginal kappa) of 0.65, and notes that tone is rarely fully written. Yoruba is a tonal language written with tone marks (grave, acute and macron) and underdots (as in ẹ, ọ, ṣ), and tweets use these marks inconsistently.

| Split | Tweets | Seen in training | Unseen |
|---|---|---|---|
| Train | 8,522 | not applicable | not applicable |
| Validation | 2,090 | 306 | 1,784 |
| Test | 4,515 | 577 | 3,938 |

The split sizes are one tweet smaller than those printed in the AfriSenti paper (8,523 training, 2,091 development and 4,516 test tweets); this study uses the sizes of the files it loaded. In the unseen test set, 1,647 tweets are positive, 1,422 neutral and 869 negative, so negative is the smallest class.

### 4.2 Overlap between training and evaluation tweets

A tweet counts as seen when it is a near copy of a training tweet. Three checks are combined: an exact match after cleaning, a match on the letters alone (with and without hashtags), and a character 3 to 5-gram TF-IDF cosine similarity of at least 0.90 with some training tweet, recomputed on the aligned text. 12.8% of the test tweets and 14.6% of the validation tweets are seen. These tweets are easy because the model has effectively memorised them: for the learned models, the score on seen test tweets is about 0.20 to 0.24 macro-F1 higher than on unseen ones, while a majority-class baseline shows only a 0.016 difference. All model selection uses the unseen validation tweets, and the headline result is the score on the unseen test tweets. The full test score is reported only for comparison.

### 4.3 Text preparation

The published test split is a cleaned version of the raw tweet text (links, retweet markers, hashtags, mentions and punctuation removed, lower case), while the training split is raw. A small search over cleaning rules found one rule that reproduces the test text exactly for 91.2% of matched pairs: lower-case the tweet, delete links, a leading retweet marker, hashtag tokens and mentions, then delete every character that is not a letter, a combining mark, a digit or a space. This rule, called text alignment, is applied to every split so that training and evaluation text have the same form. Training on aligned text was never worse than training on raw text and gained 0.026 to 0.033 macro-F1 on cleaned input. A model trained on aligned text also read raw and aligned validation tweets equally well (0.724 and 0.725).

A second step removes all tone marks and underdots (decomposing each character, dropping the combining marks and recomposing). The three problems studied later depend on flags computed per tweet:

- Code-mixing: tweets are grouped as mixed (359 in the unseen test set), English only (22) or no English cue (3,557), using a word list of English function words and Pidgin cues. This detector is a lower bound, since it misses mixing that uses no listed word.
- Negation: a tweet is flagged when it contains a Yoruba or English negation word from a fixed list (932 unseen test tweets carry a strong cue).
- Tone marks: 3,130 of the 3,938 unseen test tweets carry at least one tone mark and 808 carry none.

Checks on the raw data also found links and doubled quotation marks in some tweets, which the cleaning rule removes.

## 5. Methodology

Every model is trained on the training split, chosen on the unseen validation tweets, and scored once on the unseen test tweets; each of the three problems is handled in the model or the input and then tested as its own experiment.

### 5.1 Metrics and noise rules

The main metric is macro-F1, the mean of the F1 scores of the three classes, because the classes are unequal in size and a model that ignores negative tweets should not look good. Accuracy and per-class F1 are reported alongside it. Confidence intervals come from 1,000 bootstrap resamples of the test tweets. Two measured noise levels set the bar for claims: the standard deviation across training seeds is about 0.004, and the half-width of the test bootstrap interval is about 0.014. A difference below about 0.015 is therefore not presented as an improvement.

### 5.2 Models

- Majority class: always predicts the most frequent label, as a floor.
- Word TF-IDF with logistic regression: single words and word pairs.
- Word plus character TF-IDF with logistic regression: adds character n-grams, which helps with the inconsistent spelling of Yoruba tweets. Regularisation and n-gram settings were chosen on the unseen validation tweets.
- AfriBERTa base (about 111 million parameters, pretrained on 11 African languages including Yoruba), fine-tuned.
- AfroXLMR base (XLM-R vocabulary, adapted to 17 African languages plus Arabic, French and English), fine-tuned.

Both transformers are fine-tuned with a custom PyTorch loop: mixed-precision (fp16) arithmetic, class-weighted cross-entropy, the AdamW optimiser with linear warm-up, a maximum length of 128 tokens, batches of 32, and the epoch with the best unseen-validation macro-F1 kept. AfriBERTa was run with 4 seeds on the original text and 3 seeds on diacritic-free text. AfroXLMR was run with one seed at two learning-rate and length settings.

### 5.3 Handling the three problems

| Problem | Handled in the model or input | Tested as |
|---|---|---|
| Missing tone marks | All tone marks and underdots are removed from the input, so text with and without them looks the same | Original versus diacritic-free text, overall and on tweets that naturally have no tone marks |
| Negation | Negation scope tagging: words after a negation cue are marked, with scope length k (k = 0 means no tagging) | k varied on validation tweets, scored on tweets with negation cues |
| Code-mixing | Character n-grams for the linear model; AfroXLMR as a candidate for mixed tweets | Error rate by mixing group, and routing mixed tweets to AfroXLMR |

### 5.4 Selection rule

The deployed main model is the one with the best unseen-validation macro-F1 among those within noise of each other, which gave the word plus character logistic regression on diacritic-free text. AfriBERTa on diacritic-free text is deployed beside it as a second opinion.

## 6. Experiments and Results

The best test macro-F1 on unseen tweets is 0.732, from a word plus character logistic regression on diacritic-free text, and the fine-tuned transformers match it within noise instead of beating it.

### 6.1 Overall comparison

| Model | Input text | Macro-F1, unseen test | Note |
|---|---|---|---|
| Majority class | original | 0.197 | floor |
| Word TF-IDF, logistic regression | original | 0.708 | words and word pairs |
| Word + character TF-IDF, logistic regression | original | 0.726 |  |
| Word + character TF-IDF, logistic regression | diacritics removed | 0.732 | deployed main model; validation 0.724 |
| AfriBERTa base | original | 0.721 | mean of 4 seeds, standard deviation 0.005 |
| AfriBERTa base | diacritics removed | 0.720 | mean of 3 seeds, standard deviation 0.004 |
| AfroXLMR base | original | 0.696 | one run (5e-5, 8 epochs); the other run (3e-5, 5 epochs) scored 0.685 |

For the deployed model, accuracy is 0.746, and the F1 scores for positive, neutral and negative tweets are 0.807, 0.740 and 0.649. On the full test set, which includes tweets seen in training, its macro-F1 is 0.759, which shows how much memorised tweets inflate the score. Adding character n-grams to the word model gained 0.018, more than the noise level. The three best entries (0.732, 0.721 and 0.720) lie within 0.015 of each other, so the study does not rank them.

Precision, recall and F1 per class for the deployed model on the unseen test tweets (precision and recall computed from the confusion matrix in Section 7.1):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.834 | 0.781 | 0.807 |
| Neutral | 0.738 | 0.741 | 0.740 |
| Negative | 0.616 | 0.686 | 0.649 |

### 6.2 Tone marks and underdots

Removing all diacritics costs nothing overall: macro-F1 moves from 0.726 to 0.732 for the logistic regression and from 0.721 to 0.720 for AfriBERTa, both within noise. The benefit appears on tweets that naturally have no tone marks, where the diacritic-free models gain 0.044 (logistic regression) and 0.048 (AfriBERTa, mean of three seeds: 0.725 against 0.677) macro-F1. The input normalisation therefore removes a real weakness at no cost, and it is what the web app applies.

### 6.3 Negation

Both model families over-predict negative on tweets with negation cues. Negation scope tagging, which marks the words after a negation cue, did not repair this: it increased errors on the targeted tweets, so the final models keep k = 0 (no tagging). This is reported as a negative result.

### 6.4 Code-mixing

The error penalty for code-mixed tweets appeared on the test split but not on the validation split, so it is treated as unstable. AfroXLMR looked better than the other models on code-mixed tweets on the test split (0.775 for the 5e-5 run, 0.734 for the 3e-5 run), but on the validation split only the 3e-5 run was ahead (0.759); the 5e-5 run scored 0.712 there, below the linear model (0.729) and AfriBERTa (0.715), so the advantage is not stable. Routing mixed tweets to the 5e-5 AfroXLMR run and all others to the main model changed overall macro-F1 by -0.002 to -0.001 on validation and +0.007 to +0.009 on test, which is within noise, and it was not deployed.

### 6.5 Other experiments

- AfroXLMR: two runs were made: 3e-5 with 5 epochs (validation 0.692, test 0.685) and 5e-5 with 8 epochs (validation 0.684, test 0.696). The second run is the one reported in the tables and used for routing, although the first had the higher validation score, so the choice between them did not follow the validation rule; both runs are below the other models, so the conclusion does not change. A tokenizer explanation was not tested.
- Learning curve: macro-F1 rises by about 0.03 per doubling of training data and is still rising at the full 8,522 tweets, so more labelled data would help more than a larger model.
- Raw versus aligned text: training on aligned text was never worse and gained 0.026 to 0.033 macro-F1 on cleaned input (Section 4.3).

## 7. Error Analysis and Discussion

The two deployed models make about the same number of errors (1,001 and 1,004 of 3,938 unseen test tweets) and two thirds of those errors are shared, so most mistakes come from the tweets and not from the choice of model.

### 7.1 Which classes are confused

| True \ predicted | Positive | Neutral | Negative | Recall |
|---|---|---|---|---|
| Positive (1,647) | 1,287 | 195 | 165 | 78.1% |
| Neutral (1,422) | 162 | 1,054 | 206 | 74.1% |
| Negative (869) | 94 | 179 | 596 | 68.6% |

The table is for the logistic regression. Neutral sits at the centre of the confusion: 742 of the 1,001 errors (74%) have neutral as the true or the predicted label, so the harder question is whether a tweet carries sentiment at all. Direct flips between positive and negative make up 259 errors (26%). Negative is the weakest class (recall 68.6%, precision 61.6%), and it is over-predicted: the model says negative 967 times for 869 truly negative tweets.

### 7.2 Effect of tweet type

| Tweet group | Tweets | Error rate | 95% interval | Macro-F1 |
|---|---|---|---|---|
| Strong negation cue | 932 | 30.7% | 28.0 to 33.7 | 0.684 |
| No strong negation cue | 3,006 | 23.8% | 22.3 to 25.3 | 0.744 |
| Code-mixed | 359 | 31.5% | 26.5 to 36.5 | 0.677 |
| No English cue | 3,557 | 24.7% | 23.3 to 26.0 | 0.739 |
| No tone marks | 808 | 26.5% | 23.4 to 29.7 | 0.712 |
| With tone marks | 3,130 | 25.1% | 23.6 to 26.6 | 0.730 |

Negation is the clearest effect: its interval does not overlap the rest, and it accounts for about 64 more errors than the base rate would give, roughly 6% of all errors. The code-mixing gap is smaller, about 24 extra errors, and the earlier experiments did not reproduce it on validation, so it is suggestive only. Missing tone marks cause no penalty beyond noise, which confirms the input normalisation. English-only tweets (22) are too few to interpret. Each of the three problems explains only a small share of the errors: most errors occur in tweets with none of these features. Both flags are word-list proxies, so the true effects may be larger.

### 7.3 Shared errors and agreement

Both models are wrong on 657 tweets, the logistic regression alone on 344 and AfriBERTa alone on 347. If the models erred independently at their own rates, only about 255 tweets would be wrong for both, so the observed overlap is about 2.6 times larger. Combining the models cannot fix the shared errors. Agreement is nevertheless informative: when the two models agree (79.3% of tweets) the shared prediction is right 83.0% of the time, and when they disagree (20.7%) each is right only about 42% of the time. The disagreement set holds 470 of the logistic regression's 1,001 errors.

### 7.4 Confidence

The logistic regression is well calibrated, with an expected calibration error of 0.014: tweets it scores at 0.9 or above (1,066 tweets) are right 96.9% of the time. AfriBERTa is strongly overconfident, with an expected calibration error of 0.210: it scores 3,387 tweets (86%) at 0.9 or above and is right on only 79.2% of them. Its probabilities should therefore not be read as certainty. The web app presents them with a warning.

### 7.5 What the model has learned

The strongest word features of the logistic regression are greetings, blessings and thanks for positive (for example "e ku", "ire", "olorun", "dupe"), words of wrongdoing and conflict for negative (for example "ole", "iku", "iro", "ija", "were"), and question and reporting phrases for neutral (for example "ki ni", "pe awon", "mo wipe"). The glosses are approximate because removing diacritics merges words and were not checked by a second reader. Neutral is recognised by the absence of sentiment words, which helps explain why it is the centre of the confusion. A word-level model follows the word and not its use: tweets containing "olorun" or "eku" are called positive even when labelled negative, and tweets containing "ole" or "were" are called negative when labelled neutral.

### 7.6 Confident errors and label quality

Only 28 tweets (2.8% of the logistic regression's errors) are wrong for both models while both are at least 90% confident, spread across every direction of error. If many labels were plainly wrong, more confident disagreements with the label would be expected, so most shared errors look like genuinely ambiguous tweets: descriptive, proverbial or religious tweets that imply praise or criticism without an explicit sentiment word. A manual check of label quality was prepared but not completed, so this reading could not be tested directly (Section 9).

## 8. Deployment

The models are served by a public web app on Hugging Face Spaces where a user pastes a Yoruba tweet and gets a prediction from both deployed models.

- App: [Space link to be added]
- Code: [GitHub repository link to be added]
- Models: https://huggingface.co/Thevyne/yoruba-sentiment-models

### 8.1 Design

The app is written in Gradio. The two trained models live in a Hugging Face model repository (the logistic regression pipeline as a joblib file, 2.6 MB, and the fine-tuned AfriBERTa as a folder of about 450 MB) and the Space downloads them when it starts, so the Space itself holds only  and . A pasted tweet passes through the same preparation as the training data, text alignment and then tone-mark removal, so text typed with or without tone marks gives the same prediction. The app then shows the cleaned text the models saw, the predicted class with probabilities from the logistic regression (the main model), and the predicted class from AfriBERTa (the second opinion).

Two notes in the interface follow from the error analysis. When the models agree the app states that, in testing, about 83% of tweets with agreement were classified correctly, and when they disagree it states that each model was right only about 42% of the time on such tweets. The AfriBERTa box is labelled to warn that its probabilities are overconfident (Section 7.4).

### 8.2 Verification

The deployed logistic regression gives the same probabilities as the experiment notebook on the same sentence (0.45 neutral, 0.44 positive, 0.11 negative in the app against 0.449, 0.439 and 0.112 in the notebook), after pinning scikit-learn to version 1.6.1, the version used for training, because a model saved with one version and loaded with another can give different results. In the error-analysis notebook, the saved models reproduce the reported scores (macro-F1 0.732 and 0.726).

### 8.3 Hosting limits

The free hardware offered for this Space was ZeroGPU, so the transformer runs inside a function marked for GPU use and is subject to a small daily GPU allowance per visitor. The first request after a period of inactivity is slow because the Space must wake and load about 450 MB of weights. The app handles one tweet at a time and does not detect sarcasm.

## 9. Limitations and Future Work

The main limits are the unmeasured quality of the labels, the small gaps between the top models, and detectors for the three problems that are only word-list proxies.

### 9.1 Limitations

- Label quality was not measured. A manual check of the labels, with two readers working alone, was prepared (120 tweets: 90 model errors and 30 correct controls) but the returned sheets could not be verified as independent, so no figure from them is reported. The statement in Section 7.6 that most shared errors look ambiguous rather than mislabelled rests on confidence patterns only.
- The top models are within noise. The logistic regression (0.732) and the two AfriBERTa variants (0.720 and 0.721) differ by less than the 0.015 noise bar, so the choice of the logistic regression as the main model rests on validation score, simplicity and calibration, not on a clear win.
- AfroXLMR was run with one seed and limited tuning, and the explanation that its tokenizer suits Yoruba less well was not tested, so its weaker score (0.685 to 0.696) should not be read as a verdict on the model.
- The code-mixing detector is a lower bound, because it only sees mixing that uses listed words, and the negation detector uses a fixed word list that no Yoruba reader has verified. The effects reported in Section 7.2 may be larger than measured, and the code-mixing effect did not replicate across splits.
- Negation scope tagging did not help, so negation is handled only through the models' own representations.
- Sarcasm is out of scope. Yor-Sarc, a gold-standard Yoruba sarcasm dataset of 436 instances annotated by three native speakers (Jimoh et al., 2026), is the natural starting point for data, and contextual models such as C-Net (Jena et al., 2020) show how surrounding context can be used.
- Test tweets seen in training make up 12.8% of the test split, so the full-test score (0.759) is inflated and should not be compared with other work.
- The word glosses in Section 7.5 were not checked by a second reader.
- The deployed app depends on free hosting with a GPU allowance and a slow first request.

### 9.2 Future work

1. Complete a two-reader label check with clearer rules, and re-score the models on corrected labels. This would show how much of the error is label noise.
2. Collect more labelled Yoruba tweets. The learning curve is still rising by about 0.03 macro-F1 per doubling of the data.
3. Run more seeds and a wider search for AfroXLMR, and test whether its tokenizer explains its weaker score.
4. Replace the word-list detectors with a verified lexicon and a language-identification step for code-mixing.
5. Calibrate AfriBERTa (for example by temperature scaling) so its probabilities can be shown to users.
6. Add a sarcasm detector as a separate stage.

### 9.3 Resources used and own work

Used from others: the AfriSenti Yoruba data (CC BY 4.0), the pretrained models castorini/afriberta_base and Davlan/afro-xlmr-base, and the libraries scikit-learn, PyTorch, Transformers, Hugging Face Hub, Gradio, pandas and NumPy. Own work: the seen/unseen split of the evaluation sets, the text-alignment rule recovered from the test split, the three problem detectors and their experiments, the fine-tuning routine, the error analysis, the saved models and the web app. AI assistance was used for drafting and checking code and text; all numbers come from the notebooks.

## 10. Conclusion

For Yoruba tweet sentiment, an honest evaluation and clean input matter more than model size: a word and character logistic regression scores 0.732 macro-F1 on unseen test tweets, no worse than fine-tuned transformers, and is well calibrated.

Three results shape how the system should be used and improved. First, tweets seen in training inflate the headline score, so unseen tweets are the right basis for any claim. Second, removing tone marks and underdots is a free robustness gain, which is why the deployed app normalises its input, while negation remains the clearest weakness (30.7% error against 23.8%) and the simple scope-tagging fix did not help. Third, most errors are shared by both models and involve the neutral class, so changing the model family will not remove them. More labelled data (the learning curve is still rising), a measured estimate of label quality, and a verified negation and code-mixing lexicon are the most promising next steps. The deployed app shows both models, warns that the transformer's probabilities are overconfident, and reports how often agreement between the models has been correct.

## 11. References

- Adelani, D. I. (2022). Natural language processing for African languages. Doctoral thesis, Saarland University, Saarbrücken. 
- Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In Proceedings of the 29th International Conference on Computational Linguistics (COLING), pp. 4336–4349. 
- Das, S. R., and Chen, M. Y. (2001). Yahoo! for Amazon: Sentiment parsing from small talk on the web. In Proceedings of the European Finance Association (EFA) Annual Conference 2001. Journal version: 
- Jena, A. K., Sinha, A., and Agarwal, R. (2020). C-Net: Contextual network for sarcasm detection. In Proceedings of the Second Workshop on Figurative Language Processing, pp. 61–66. 
- Jimoh, T. A., De Wille, T., and Nikolov, N. S. (2026). Yor-Sarc: A gold-standard dataset for sarcasm detection in a low-resource African language. 
- Joshi, P., Santy, S., Budhiraja, A., Bali, K., and Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. 
- Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter sentiment corpus for multilingual sentiment analysis. In Proceedings of the 13th Language Resources and Evaluation Conference (LREC), pp. 590–602. 
- Muhammad, S. H., Abdulmumin, I., Ayele, A. A., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 13968–13981. 
- Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resource languages. 
- Pang, B., Lee, L., and Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. In Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 79–86. 
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of EMNLP 2020: System Demonstrations, pp. 38–45.
- Abid, A., et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.
- AfriSenti Twitter sentiment dataset, `shmuhammad/AfriSenti-twitter-sentiment`, Hugging Face Datasets (CC BY 4.0).
- Models: `castorini/afriberta_base` and `Davlan/afro-xlmr-base`, Hugging Face Hub.
