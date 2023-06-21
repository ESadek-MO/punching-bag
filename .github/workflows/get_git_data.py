# pip install PyGithub

import csv
import datetime
from github import Github
import github
from io import StringIO
import json
import logging
import os
from pathlib import Path
from pprint import pprint
import requests
from requests.adapters import HTTPAdapter
import time
from urllib3.util.retry import Retry

retries = Retry(
    total=5,
    backoff_factor=1,
    # see https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
    status_forcelist=[500, 502, 503, 504, 202],
)

# This disables the ""InsecureRequestWarning" when using requests for https
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# set logging level (NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)

# configure all the end points to use for gathering the stats
# set them to "" if you want to not include them
RELEASES_SSS_CSV_URL = "https://www-avd/sci/software_stack/_static/releases.csv"

# See https://metoffice.atlassian.net/wiki/spaces/AVD/pages/2931196057/NG-VAT+Releases
RELEASES_OTHER = [
    ["NG-VAT", "v0.1", "2020-09-22 0:00:00"],
    ["NG-VAT", "v0.2", "2021-05-03 0:00:00"],
    ["NG-VAT", "v0.2 (update)", "2021-05-13 0:00:00"],
    ["NG-VAT", "v0.2 (update)", "2021-06-17 0:00:00"],
    ["NG-VAT", "v0.2 (update)", "2021-06-22 0:00:00"],
    ["NG-VAT", "v0.3", "2021-12-14 0:00:00"],
    ["NG-VAT", "v1.0", "2022-04-04 0:00:00"],
]

DATA_DIR = "data"
IMAGES_DIR = "data"

# Date format to use when output dates in the csv
DATE_STR_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def autolog_debug(message):
    # Get the previous frame in the stack, otherwise it would
    # be this function.
    func = inspect.currentframe().f_back.f_code
    head, file_name = os.path.split(func.co_filename)
    # Dump the message + the name of this function to the log.
    logging.debug(
        "%s in %s:%i \n    --> %s"
        % (func.co_name, file_name, func.co_firstlineno, message)
    )


def autolog_info(message):
    # Dump the message + the name of this function to the log.
    logging.info(" --> {}".format(message))


def check_dir(d):
    if not os.path.isdir(d):
        autolog_info(f"Creating directory: {d}")
        os.mkdir(d)


def get_token(filename):
    """load the github api token from disk"""

    token_file = Path(filename)

    if not token_file.exists():
        raise IOError(
            "Please create a token at github.com, and save "
            + "it in {}".format(token_file)
        )

    token = token_file.read_text().strip()

    return token


def get_releases(g, repo_list):
    """Retrieve releases innfo and write to csv

    Will query the github api to releases for the list of
    repos provided.  Will also append some static csv entries
    if needed.

    Note
    ----
    Command line equivalent:
    curl -i "https://api.github.com/repos/scitools/iris/releases"

    https://pygithub.readthedocs.io/en/latest/github_objects/GitRelease.html#github.GitRelease.GitRelease
    """

    # setup the csv file to write
    check_dir(DATA_DIR)
    csv_output = os.path.join(DATA_DIR, "releases.csv")
    fh = open(csv_output, "w", newline="")

    csv_out = csv.writer(fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    csv_header = [
        "repo_name",
        "release_name",
        "published_at",
        "repo_url",
        "release_url",
    ]
    csv_out.writerow(csv_header)

    # write the 'other' release info
    if RELEASES_OTHER:
        csv_out.writerows(RELEASES_OTHER)

    # write the SSS release info
    if RELEASES_SSS_CSV_URL:
        autolog_info(f"Internal: Fetch SSS releases: {RELEASES_SSS_CSV_URL}")

        r = requests.get(RELEASES_SSS_CSV_URL, verify=False)
        r.raise_for_status()

        content = r.text
        reader = csv.reader(content.split("\n"), delimiter=",")

        for i, row in enumerate(reader):
            # handle header and blank link at end of file
            if i > 0 and row:
                csv_out.writerow(
                    [
                        "Scientific Software Stack",
                        row[1],
                        row[0],
                        "https://github.com/MetOffice/ssstack/tree/main/environments",
                        "",
                    ]
                )

    # now fetch the github release info
    if repo_list:
        autolog_info(f"GitHub API: Releases, project to process: {len(repo_list)}")

        for repo_name in repo_list:
            repo = g.get_repo(repo_name)
            releases = repo.get_releases()
            total_releases = releases.totalCount

            autolog_info(f"Found {total_releases} for project {repo_name}")

            for release in releases:
                csv_out.writerow(
                    [
                        repo_name,
                        release.title,
                        release.published_at,
                        repo.html_url,
                        release.html_url,
                    ]
                )

    fh.close()


def get_pulls(g, repo_name):
    # curl -i "https://api.github.com/repos/scitools/iris/pulls"
    repo = g.get_repo(repo_name)

    # https://pygithub.readthedocs.io/en/latest/github_objects/PullRequest.html#github.PullRequest.PullRequest
    pulls = repo.get_pulls(state="all")
    total_pulls = pulls.totalCount

    check_dir(DATA_DIR)
    csv_output = os.path.join(DATA_DIR, repo_name.replace(os.sep, "-") + "-pulls.csv")

    fh = open(csv_output, "w", newline="")
    csv_out = csv.writer(fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    csv_header = [
        "number",
        "html_url",
        "pr_state",
        "title",
        "user",
        "created_at",
        "updated_at",
        "merged_at",
        "closed_at",
    ]
    csv_out.writerow(csv_header)

    autolog_info(f"GitHub API: Pulls to process: {total_pulls}")
    message_frequency = 250

    for i, pull in enumerate(pulls):
        csv_out.writerow(
            [
                pull.number,
                pull.html_url,
                pull.state,
                pull.title,
                pull.user.login,
                pull.created_at,
                pull.updated_at,
                pull.merged_at,
                pull.closed_at,
            ]
        )

        if (i + 1) % message_frequency == 0:
            autolog_info(f"Processed {i+1}/{total_pulls} ...")

    fh.close()
    autolog_info(f"Successfully processed {i+1}/{total_pulls} pulls")


def get_issues(g, repo_name):
    # curl -i "https://api.github.com/repos/scitools/iris/issues"
    repo = g.get_repo(repo_name)

    # https://pygithub.readthedocs.io/en/latest/github_objects/Issue.html#github.Issue.Issue
    issues = repo.get_issues(state="all")
    total_issues = issues.totalCount

    check_dir(DATA_DIR)
    csv_output = os.path.join(DATA_DIR, repo_name.replace(os.sep, "-") + "-issues.csv")

    fh = open(csv_output, "w", newline="")
    csv_out = csv.writer(fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    csv_header = [
        "number",
        "html_url",
        "issue_state",
        "title",
        "user",
        "created_at",
        "closed_at",
    ]
    csv_out.writerow(csv_header)

    autolog_info(
        f"GitHub API: Issues (including pull requests) to process: {total_issues}"
    )
    message_frequency = 250

    for i, issue in enumerate(issues):
        # do not use issue.pull_request as this fires another API call to GitHub (slow)
        if issue._pull_request is github.GithubObject.NotSet:
            csv_out.writerow(
                [
                    issue.number,
                    issue.html_url,
                    issue.state,
                    issue.title,
                    issue.user.login,
                    issue.created_at,
                    issue.closed_at,
                ]
            )

        if (i + 1) % message_frequency == 0:
            autolog_info(f"Processed {i+1}/{total_issues} ...")

    fh.close()
    autolog_info(f"Successfully processed {i+1}/{total_issues} issues")


def get_commits(repo_name, csv_output):
    # curl -i "https://api.github.com/repos/stats/contributors"
    repo = g.get_repo(repo_name)

    spinner = Counter("Commits: ")

    # https://pygithub.readthedocs.io/en/latest/github_objects/Commit.html#github.Commit.Commit
    commits = repo.get_commits()
    total_commits = commits.totalCount

    fh = open(csv_output, "w")
    fh.write("author,start_of_week,additions,deletions,commits")

    autolog_info(f"GitHub API: Commits to process: {total_commits}")
    message_frequency = 250

    for i, commit in enumerate(commits):
        spinner.next()

        # accessing commit.stats.* forces another call to the GitHub API
        # this it is slow.
        fh.write(
            f"{commit.author},{commit.author},{commit.stats.additions},"
            f"{commit.stats.deletions},{commit.stats.deletions},{commit.stats.total},"
            f"{os.linesep}"
        )

    fh.close()
    print()


def get_commits_direct(repo_name, token):
    # curl -i "https://api.github.com/repos/stats/contributors"

    query_url = f"https://api.github.com/repos/{repo_name}/stats/contributors"
    headers = {"Authorization": f"token {token}"}

    autolog_info(f"GitHub API: Commits fetch direct: {query_url}")

    # This request can sometimes fail with a http request code of 202.
    # Lets try a few times before raising an exception.
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    r = sess.get(query_url, headers=headers)
    r.raise_for_status()

    if r.status_code == 202:
        raise Exception(f"Commits request failed, HTTP status code = {r.status_code}.")
    else:
        autolog_info(f"GitHub API: HTTP status code = {r.status_code}")

    check_dir(DATA_DIR)
    json_output = os.path.join(
        DATA_DIR, repo_name.replace(os.sep, "-") + "-commits.json"
    )
    fh = open(json_output, "w")
    json.dump(r.json(), fh)
    fh.close()

    autolog_info(f"GitHub API: Commits creating csv file")

    csv_output = os.path.join(DATA_DIR, repo_name.replace(os.sep, "-") + "-commits.csv")
    fh = open(csv_output, "w", newline="")
    csv_out = csv.writer(fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    csv_header = [
        "author",
        "week",
        "additions",
        "deletions",
        "commits",
        "avatar_url",
    ]
    csv_out.writerow(csv_header)

    fh = open(csv_output, "w")
    for item in r.json():
        for week in item["weeks"]:
            csv_out.writerow(
                [
                    item["author"]["login"],
                    week["w"],
                    week["a"],
                    week["d"],
                    week["c"],
                    item["author"]["avatar_url"],
                ]
            )

    fh.close()
