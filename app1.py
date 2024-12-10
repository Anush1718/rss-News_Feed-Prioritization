from flask import Flask, jsonify, render_template
import feedparser
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news.db'
db = SQLAlchemy(app)

# Model to represent news articles in the database
class NewsArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    link = db.Column(db.String(500), nullable=False, unique=True)
    source = db.Column(db.String(100), nullable=False)
    published_date = db.Column(db.String(100), nullable=True)
    frequency = db.Column(db.Integer, default=1)  # Tracks the number of times an article appears
    description = db.Column(db.String(500), nullable=True)  # New column for description

    def __repr__(self):
        return f'<NewsArticle {self.title}>'

# Create the database tables (if not already created)
with app.app_context():
    db.create_all()

RSS_FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://mashable.com/feeds/rss/all",
    "https://gizmodo.com/feed",
    "https://www.cnet.com/rss/news/",
    "https://feeds.feedburner.com/venturebeat/SZYF",

]

def fetch_rss_feed(url):
    try:
        feed = feedparser.parse(url)
        return feed['entries']
    except Exception as e:
        return []

def save_article(entry, source):
    existing_article = NewsArticle.query.filter_by(link=entry.link).first()
    if existing_article:
        return
    else:
        # If it's a new article, add it to the database
        article = NewsArticle(
            title=entry.title,
            link=entry.link,
            source=source,
            published_date=entry.published if hasattr(entry, 'published') else None,
            description=entry.summary if hasattr(entry, 'summary') else None  # Add description
        )
        db.session.add(article)
        db.session.commit()

def calculate_similarity(titles):
    vectorizer = TfidfVectorizer().fit_transform(titles)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    return cosine_matrix

@app.route('/')
@app.route('/')
def index():
    # Clear previous articles from the database
    NewsArticle.query.delete()  # This will delete all existing articles
    db.session.commit()  # Commit the changes to the database

    all_sources = {url: fetch_rss_feed(url) for url in RSS_FEEDS}

    # Save new articles to the database
    for source, entries in all_sources.items():
        for entry in entries:
            save_article(entry, source)

    # Retrieve all articles from the database
    articles = NewsArticle.query.all()

    # Calculate similarity matrix for titles only
    titles = [article.title for article in articles]
    title_similarity_matrix = calculate_similarity(titles)

    # Initialize clusters for grouping similar articles
    unique_articles = []

    for i, article_i in enumerate(articles):
        # Initialize frequency for each article
        article_i.frequency = 1  # Start with frequency of 1
        
        for j in range(len(titles)):
            if i != j:  # Avoid self-comparison
                title_similarity = title_similarity_matrix[i][j]
                
                if title_similarity > 0.5:  # Assuming 0.5 is the threshold
                    article_i.frequency += 1  # Increment frequency for similar titles
        
        unique_articles.append(article_i)

    # Sort unique articles by frequency in descending order
    sorted_unique_articles = sorted(unique_articles, key=lambda x: x.frequency, reverse=True)

    # Debug output to verify frequencies
    for article in sorted_unique_articles:
        print(f"Article: {article.title}, Frequency: {article.frequency}")

    return render_template('index.html', articles=sorted_unique_articles)

if __name__ == '__main__':
    app.run(debug=True)
