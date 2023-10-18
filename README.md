# balancr

A Flask app that computes topics from news articles. It uses [BERTopic](https://maartengr.github.io/BERTopic/index.html) for topic modeling.

Features:

- Choose a time period and what parts of an article to include (headline, teaser, etc.) and compute most dominant topics for this period
- Visualize top-ten topics and their representation as well as how many articles of a given topic have been published in each medium
- Generate summaries for each topic using an LLM
- If a topic isn't concise enough, you can use the drill down function to compute child-topics from a given topic

Demonstration (video is sped-up):

https://github.com/karimdabbouz/balancr/assets/122094147/bfb97495-b185-40de-8644-4c86fe0826a0


Future features:

- Improve LLM summaries and labels
- Pre-train topic clusters (possibly schedule them so they run automatically)
- Expose routes as API endpoints
- Build a proper JS frontend
- Deploy a demo
- Add more news sources