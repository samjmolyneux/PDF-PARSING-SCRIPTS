# Integration-test PDFs

These two unchanged papers are copied from
[Flowde's CONSORT demo](https://github.com/EPPI-Centre/Flowde/tree/c92a183465f10090a2a9b9e1b5b2e381998c9123/data/consort-demo).
Their own Creative Commons licences apply to the PDFs. Attribution and source
metadata are retained in [manifest.json](manifest.json) and inside the PDFs.

| PDF               | Attribution and original article                                                                                                                                                                                                                           | Licence                                                   | Pages |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----- |
| `Young_2008.pdf`  | Young, Girgis, Bruce, Hobbs and Ward (2008). [Acceptability and effectiveness of opportunistic referral of smokers to telephone cessation advice from a nurse: a randomised trial in Australian general practice](https://doi.org/10.1186/1471-2296-9-16). | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | 10    |
| `Vander_2016.pdf` | Vander Weg and colleagues (2016). [An individually-tailored smoking cessation intervention for rural Veterans: a pilot randomized trial](https://doi.org/10.1186/s12889-016-3493-z).                                                                       | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 11    |

The fixtures include prose, tables and CONSORT figures. The integration tests
check selected text near the beginning, middle and end of each paper, page counts,
and one row from Table 1 (Young: page 6; Vander: page 7). Text checks tolerate
spacing, punctuation and HTML styling differences. Table checks preserve column
order. The PDFs are kept whole so these tests also exercise document batching.
