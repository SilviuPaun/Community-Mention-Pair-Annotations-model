## Community Mention Pair Annotations (CommunityMPA) model

This is a variational inference implementation of the *Community Mention Pair Annotations* (CommunityMPA) model presented in:

Silviu Paun, Juntao Yu, Jon Chamberlain, Udo Kruschwitz, Massimo Poesio (2019). **A Mention-Pair Model of Annotation with Nonparametric User Communities**.

The model is a partially pooled nonparametric extension of the MPA model from:

Silviu Paun, Jon Chamberlain, Udo Kruschwitz, Juntao Yu, Massimo Poesio (2018). **A Probabilistic Annotation Model for Crowdsourcing Coreference**. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

The code is written in JAVA and requires the Apache Commons Mathematics library. The input file (e.g., example.csv) assumes the following structure:

```
mention_id,annotator_id,gold,annotation
ne9399,annotator1,DO(ne9398),DO(ne9398)
ne9399,annotator2,DO(ne9398),DO(ne9395)
ne9399,annotator3,DO(ne9398),DO(ne9398)
ne9399,annotator4,DO(ne9398),DO(ne9396)
...
```

The header describes the id of the mention, the id of the annotator, the gold (expert) label and the annotation label provided by the annotator. The code will automatically extract the class from the annotation label (e.g.: DO).

The model is set, by default, to use for initialization the user profiles estimated by MPA (mpa-profiles-example.csv).

Running the code produces posterior point estimates for all the model parameters. The output is set to show the accuracy of the inferred mention pairs against the gold standard and the most prevalent annotator community profiles. It also includes the accuracy of a majority vote baseline, computed over 10 random rounds of splitting ties.
