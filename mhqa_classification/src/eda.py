import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


def plot_domain_distribution(df):
    domain_counts = df['domain'].value_counts()

    plt.figure(figsize=(8,8))
    plt.pie(domain_counts,
            labels=domain_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            explode=[0.1 if i == 0 else 0 for i in range(len(domain_counts))],
            shadow=True)
    plt.title("Domain Distribution")
    plt.show()


def plot_top_words(df):
    text = " ".join(df['clean_text'])
    words = Counter(text.split()).most_common(20)

    w = [i[0] for i in words]
    c = [i[1] for i in words]

    plt.figure(figsize=(10,5))
    plt.bar(w, c)
    plt.xticks(rotation=45)
    plt.title("Top Words")
    plt.show()


def plot_wordcloud(df):
    text = " ".join(df['clean_text'])
    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)
    plt.axis('off')
    plt.show()