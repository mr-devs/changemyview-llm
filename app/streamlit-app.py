"""
CMV AI Persuasion Companion

This Streamlit app fetches submissions from the r/changemyview subreddit,
analyzes them using OpenAI's GPT model, and generates counter-arguments.
Users can view analyses, toggle their visibility, and optionally post
counter-arguments directly to Reddit.

The app uses PRAW for Reddit API interactions and OpenAI's API for text analysis.
Streamlit is used for the web interface and session state management.

Author: Matthew R. DeVerna
"""

import json
import os
import praw
import time

import streamlit as st

from openai import OpenAI
from pydantic import BaseModel

FETCH_COOLDOWN = 60


# Build a structured JSON model for
class CMV_argument(BaseModel):
    main_position: str
    rationale: list[str]


# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    password=os.environ.get("REDDIT_PASSWORD"),
    user_agent="python:changemyview_llm:v1.0 (by /u/double-o-ai-science)",
    username=os.environ.get("REDDIT_USERNAME"),
)


@st.cache_data(ttl=3600)
def get_cmv_submissions(sort_by="top", time_filter="all", limit=5):
    """
    Fetch submissions from r/changemyview subreddit.

    Args:
    sort_by (str): Method to sort submissions (top, new, hot, rising)
    time_filter (str): Time period for top submissions (day, week, month, year, all)
    limit (int): Number of submissions to fetch

    Returns:
    list: List of PRAW submission objects
    """
    subreddit = reddit.subreddit("changemyview")
    if sort_by == "new":
        return list(subreddit.new(limit=limit))
    elif sort_by == "hot":
        return list(subreddit.hot(limit=limit))
    elif sort_by == "rising":
        return list(subreddit.rising(limit=limit))
    elif sort_by == "top":
        return list(subreddit.top(time_filter=time_filter, limit=limit))
    else:
        return list(subreddit.top(time_filter=time_filter, limit=limit))


def extract_main_argument(_submission):
    """
    Extract the main argument and rationale from a CMV submission using OpenAI's GPT model.

    Args:
    _submission (praw.models.Submission): Reddit submission object

    Returns:
    dict: Analysis containing main_position and rationale
    """
    title = _submission.title
    text = _submission.selftext

    system_content = """
    You are a helpful assistant. 
    You will be presented with a post from the subreddit r/changemyview and your
    task is to extract the main argument of the poster, as well as the key rationale
    that they feel supports their position. 
    Return your response in the following JSON format:
    {
        "main_position": "The main argument of the poster",
        "rationale": ["Point 1", "Point 2", "Point 3"]
    }
    """
    user_content = f"TITLE: {title}.\nTEXT: {text}"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        response_format=CMV_argument,
        temperature=0,
    )

    try:
        analysis = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error("Failed to parse the analysis response. Using a default structure.")
        analysis = {
            "main_position": "Could not extract main position",
            "rationale": ["Could not extract rationale"],
        }

    return analysis


def generate_counter_argument(analysis):
    """
    Generate a counter-argument based on the analysis of a CMV submission.

    Args:
    analysis (dict): Analysis containing main_position and rationale

    Returns:
    str: Generated counter-argument
    """
    rationale_str = "\n".join(
        [f"{i+1}. {r}" for i, r in enumerate(analysis["rationale"])]
    )

    system_content = """
    You are a helpful assistant. 
    You will be presented with an argument from the subreddit r/changemyview along
    with the central rationale presented to support that argument.
    Your task is to be extremely persuasive and argue against that position.
    Be polite but make sure to address each point of rationale to counter the main argument.
    Use evidence-based arguments as much as possible and provide realistic alternatives.
    Structure and style your response like it is a post for the r/changemyview subreddit.
    """
    user_content = (
        f"MAIN ARGUMENT: {analysis['main_position']}.\nRATIONALE: {rationale_str}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def analyze_submission(_submission):
    """
    Analyze a CMV submission using OpenAI's GPT model.

    Args:
    _submission (praw.models.Submission): Reddit submission object

    Returns:
    tuple: (analysis dict, counter_argument string)
    """
    analysis = extract_main_argument(_submission)
    counter_argument = generate_counter_argument(analysis)
    return analysis, counter_argument


def post_to_reddit(submission, counter_argument):
    """
    Post a counter-argument as a reply to a Reddit submission.

    Args:
    submission (praw.models.Submission): Reddit submission object
    counter_argument (str): The counter-argument to post

    Returns:
    bool: True if posted successfully, False otherwise
    """
    try:
        submission.reply(counter_argument)
        return True
    except Exception as e:
        st.error(f"Failed to post comment: {str(e)}")
        return False


def main():
    """
    Main function to run the Streamlit app.
    """
    global client
    st.title("AI Persuasion Companion for CMV")

    # Add app description
    st.markdown(
        """
    Welcome to the AI Persuasion Companion for Change My View (CMV)!

    This app fetches recent submissions from the [r/changemyview](https://www.reddit.com/r/changemyview/) subreddit, analyzes them using AI, and generates counter-arguments. You can:
    
    1. View top posts from CMV
    2. Analyze the main arguments and rationale
    3. See AI-generated counter-arguments
    
    To get started, enter your OpenAI API key below and click 'Fetch New Submissions'.
    """
    )

    # OpenAI API key input
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Initialize OpenAI client if API key is provided
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        st.success("OpenAI client initialized successfully!")
    else:
        st.warning("Please provide the OpenAI API key to proceed.")
        return

    # Create container for sort options
    sort_container = st.container()

    # Create two columns within the container
    col1, col2, col3 = sort_container.columns(3)

    with col1:
        fetch_items = [3, 5, 10]
        limit = st.selectbox("Number of posts to fetch", fetch_items)

    with col2:
        sort_options = ["top", "new", "hot", "rising"]
        selected_sort = st.selectbox("Sort submissions by:", sort_options, index=0)

    with col3:
        if selected_sort == "top":
            time_filters = ["day", "week", "month", "year", "all"]
            selected_time = st.selectbox("Time period:", time_filters, index=0)
        else:
            selected_time = "all"  # Default value when not using "top" sort
            st.empty()  # Placeholder to maintain layout

    # Add this before the "Fetch New Submissions" button
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = 0

    # Replace the existing "Fetch New Submissions" button code with this:
    fetch_button = st.button(
        "Fetch New Submissions",
        help="Please wait 60 seconds between fetching posts.",
    )
    if fetch_button:
        current_time = time.time()
        if current_time - st.session_state.last_fetch_time >= FETCH_COOLDOWN:
            with st.spinner("Fetching submissions..."):
                submissions = get_cmv_submissions(
                    sort_by=selected_sort, time_filter=selected_time, limit=limit
                )
            st.session_state.submissions = submissions
            st.session_state.last_fetch_time = current_time
        else:
            remaining_time = int(
                FETCH_COOLDOWN - (current_time - st.session_state.last_fetch_time)
            )
            st.warning(f"Please wait {remaining_time} seconds before fetching again.")

    # Display and process submissions
    if "submissions" in st.session_state:
        for submission in st.session_state.submissions:
            # Initialize session state for each submission
            if submission.id not in st.session_state:
                st.session_state[submission.id] = {
                    "analyzed": False,
                    "visible": False,
                    "analysis": None,
                    "counter_argument": None,
                }

            # Determine button text based on visibility state
            button_text = (
                "Hide" if st.session_state[submission.id]["visible"] else "Analyze"
            )
            button_label = f"{button_text}: {submission.title}"

            # Create the button with improved readability
            button_key = f"toggle_{submission.id}"
            if st.button(button_label, key=button_key):
                is_analyzed = st.session_state[submission.id]["analyzed"]
                if not is_analyzed:
                    # Analyze the submission if not already done
                    with st.spinner("Analyzing submission..."):
                        analysis, counter_argument = analyze_submission(submission)
                    st.session_state[submission.id]["analysis"] = analysis
                    st.session_state[submission.id][
                        "counter_argument"
                    ] = counter_argument
                    st.session_state[submission.id]["analyzed"] = True

                # Toggle visibility
                st.session_state[submission.id]["visible"] = not st.session_state[
                    submission.id
                ]["visible"]

                # Force a rerun to update the button text immediately
                st.rerun()

            # Display analysis and counter-argument if visible
            if st.session_state[submission.id]["visible"]:
                analysis = st.session_state[submission.id]["analysis"]
                counter_argument = st.session_state[submission.id]["counter_argument"]

                # Display original submission
                st.subheader("Original Submission")
                st.write(f"**Title:** {submission.title}")
                st.write(f"**Text:** {submission.selftext}")

                # Display analysis and counter-argument only if analyzed
                if st.session_state[submission.id]["analyzed"]:
                    # Display analysis
                    st.subheader("Analysis")
                    st.write(f"**Main Position:** {analysis['main_position']}")
                    st.write("**Rationale:**")
                    for i, point in enumerate(analysis["rationale"], 1):
                        st.write(f"{i}. {point}")

                    # Display counter-argument
                    st.subheader("Counter Argument")
                    st.write(counter_argument)


if __name__ == "__main__":
    main()
