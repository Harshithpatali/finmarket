import argparse
from src.data.news import search_gnews
from src.news.features import aggregate_news
parser=argparse.ArgumentParser(); parser.add_argument("--ticker",required=True); parser.add_argument("--company",required=True); args=parser.parse_args()
articles=search_gnews(args.company,max_results=10,days=3)
print(f"Articles: {len(articles)}")
for a in articles: print("\n",a.get("publishedAt"),"|",a.get("title"),"\n",a.get("url"))
print("\nAggregated:",aggregate_news(articles))
