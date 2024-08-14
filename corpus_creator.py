import re
from convokit import Corpus, Utterance
import logging
import pandas as pd
import textstat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_comments(corpus: Corpus) -> None:
    """
    Remove text within square brackets, emojis, and everything after the word "EDIT" from all utterances in the corpus.
    """
    bracket_pattern = re.compile(r'\[.*?\]')
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    edit_pattern = re.compile(r'EDIT.*', re.IGNORECASE)

    for utt in corpus.iter_utterances():
        if utt.text is not None:
            new_text = re.sub(bracket_pattern, '', utt.text)
            new_text = re.sub(emoji_pattern, '', new_text)
            new_text = re.sub(edit_pattern, '', new_text).strip()
            utt.text = new_text


def corpus_format_to_csv(corpus_names: list[str], output_filename: str = "data.csv",
                         num_of_samples_for_corpus: int = 100) -> None:
    """
    Process the given corpora to extract top-level comments and their earliest replies.
    Save the results to a single CSV file.

    Parameters:
        corpus_names (list[str]): List of corpus file names to process.
        output_filename (str): The name of the output CSV file.
        num_of_samples_for_corpus (int): The number of samples to select from each subreddit based on the highest Flesch Reading Ease score.
    """
    corpus_data = []

    for corpus_name in corpus_names:
        logging.info(f"Processing corpus: {corpus_name}")
        try:
            corpus = Corpus(filename=corpus_name)
            preprocess_comments(corpus)
        except Exception as e:
            logging.error(f"Failed to load corpus {corpus_name}: {e}")
            continue
        logging.info(f"Loaded Corpus: {corpus_name}")

        for convo in corpus.iter_conversations():
            for utt_id in convo._utterance_ids:
                utt = corpus.get_utterance(utt_id)
                if not utt or not utt.text:
                    continue

                if utt.reply_to is None and utt.text.strip():
                    top_level_entry = {
                        "subreddit_name": convo.meta.get("subreddit", ""),
                        "conversation_title": convo.meta.get("title", ""),
                        "top_level_text": utt.text.strip(),
                        "reply_text": ""
                    }
                    earliest_reply = earliest_comment(convo, corpus, utt)

                    if earliest_reply:
                        top_level_entry["reply_text"] = earliest_reply.text.strip()

                    corpus_data.append(top_level_entry)

    df = pd.DataFrame(corpus_data)
    df = filter_df(df, num_of_samples_for_corpus)
    df.to_csv(output_filename, index=False, escapechar='\\')
    logging.info(f"Data saved to {output_filename}")


def earliest_comment(convo, corpus, top_level_comment) -> Utterance:
    # Look for the earliest reply to this top-level comment
    earliest_reply = None
    earliest_timestamp = float('inf')

    for reply_utt_id in convo._utterance_ids:
        reply_utt = corpus.get_utterance(reply_utt_id)
        if reply_utt and reply_utt.reply_to == top_level_comment.id and reply_utt.text and reply_utt.text.strip():
            if reply_utt.timestamp < earliest_timestamp:
                earliest_reply = reply_utt
                earliest_timestamp = reply_utt.timestamp

    return earliest_reply


def is_size(text, lower_limit=15, upper_limit=100):
    """
    Check if the text length (in words) is within the specified limits.

    Parameters:
        text (str): The text to check.
        lower_limit (int): The minimum number of words.
        upper_limit (int): The maximum number of words.

    Returns:
        bool: True if the text length is outside the limits, False otherwise.
    """
    if pd.isna(text):
        return True
    size = len(str(text).split())
    return size >= lower_limit and size <= upper_limit


def filter_df(df, num_of_samples_for_corpus):
    """
    Filter the DataFrame to remove rows with links, rows with text length outside the specified limits,
    and select the top num_of_samples_for_corpus comments for each subreddit based on the highest Flesch Reading Ease score.

    Parameters:
        df (DataFrame): The DataFrame to filter.
        num_of_samples_for_corpus (int): The number of samples to select from each subreddit.

    Returns:
        DataFrame: The filtered DataFrame.
    """
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[~df.apply(lambda row: row.astype(str).str.contains(r'http|www\.', case=False).any(), axis=1)]
    columns = ["top_level_text", "reply_text"]
    df = df[~df[columns].applymap(lambda x: not is_size(x)).any(axis=1)]

    # Calculate Flesch Reading Ease score and filter the top num_of_samples_for_corpus for each subreddit
    df['flesch_reading_ease'] = df['top_level_text'].apply(textstat.flesch_reading_ease)
    df = df.sort_values(by='flesch_reading_ease', ascending=False)
    df = df.groupby('subreddit_name').head(num_of_samples_for_corpus)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['flesch_reading_ease'], inplace=True)  # Remove the temporary column
    return df


if __name__ == '__main__':
    pass
    # Explanations:
    # To create Data,Have the corpus in your local and write their names in the list and run.
    corpus_names = []
    corpus_format_to_csv(corpus_names=corpus_names)
