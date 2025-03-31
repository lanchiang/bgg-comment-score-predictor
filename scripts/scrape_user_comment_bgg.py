from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright


def get_top_X_game_ids(num_games: int) -> list:
    """Get top X game IDs from BGG."""
    url = "https://boardgamegeek.com/browse/boardgame"

    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    if response.status_code != 200:
        raise Exception(f"Failed to get top {str(num_games)} game IDs from BGG. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    game_ids = []

    for link in soup.select("#collectionitems tr[id^='row_'] td[id^='CEcell_objectname'] a[href^='/boardgame/']"):
        game_id = link.get('href').split('/', maxsplit=2)[-1]
        game_ids.append(game_id)

        if len(game_ids) >= num_games:
            break

    return game_ids


def get_game_comments(game_id: str) -> list:
    """Get comments for a game from BGG."""

    results = []

    page_id = 1
    last_processed_page_id = None
    while True:
        url = f"https://boardgamegeek.com/boardgame/{game_id}/ratings?pageid={page_id}&comment=1"  # Comment=1 stands for ratings with comments

        comments = None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)

            content = page.content()

            soup = BeautifulSoup(content, 'html.parser')

            summary_items = soup.select('li.summary-item.summary-rating-item.ng-scope')

            for summary_item in summary_items:
                # extract the rating score
                rating_element = summary_item.select_one('div.summary-item-callout')
                rating_score = rating_element.text.strip() if rating_element else None

                # extract the comment text
                # comment_element = summary_item.select_one('div.comment-body.summary-item-body')
                comment_element = summary_item.select_one('p.mb-0.ng-binding.ng-scope')
                comment_text = comment_element.text.strip() if comment_element else None

                if rating_score and comment_text:
                    results.append({
                        'rating': rating_score,
                        'comment': comment_text
                    })

            # comment_selector = 'li.summary-item.summary-item-comments.ng-scope'

            # page.wait_for_selector(comment_selector, timeout=0)
            # comments = [el.inner_text() for el in page.query_selector_all(comment_selector)]

            browser.close()

        # response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

        # parsed_url = urlparse(response.url)
        # query_params = parse_qs(parsed_url.query)
        # final_pageid = query_params.get('pageid', [None])[0]

        # if final_pageid == last_processed_page_id:
        #     break

        # if response.status_code != 200:
        #     raise Exception(f"Failed to get comments for game ID {game_id} from BGG. Status code: {response.status_code}")
        
        # soup = BeautifulSoup(response.text, 'html.parser')
        
        # summary_items = soup.select('li.summary-item.summary-item-comments.ng-scope')

        page_id += 1


if __name__ == '__main__':
    num_games = 5
    game_ids = get_top_X_game_ids(num_games)
    print(game_ids)

    for game_id in game_ids:
        comments = get_game_comments(game_id)
        print(comments)
        print()