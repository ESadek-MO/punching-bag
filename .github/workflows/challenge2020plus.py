import get_git_data as gd
from github import Github
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import datapane as dp
import os
import numpy as np
import datetime as datetime
import time
import requests

# useful for testing the report only
QUERY_GITHUB = True
INFO = True
DEBUG = False

REPORT_HTML = "challenge2020plus.html"
TOKEN_FILE = "github-token.txt"

# repos to query for release info on Github
REPOS = [
    "scitools/iris",
    "scitools/iris-test-data",
    "scitools/iris-grib",
    "scitools/cf-units",
    "scitools/nc-time-axis",
    "scitools/tephi",
    "scitools/iris-agg-regrid",
    "scitools/python-stratify",
    "scitools-incubator/iris-ugrid",
    "scitools-incubator/iris-esmf-regrid",
    "sciTools/workflows",
    "bjlittle/geovista",
    "bjlittle/geovista-data",
    "bjlittle/geovista-media",
    "bjlittle/geovista-slam",
    "pp-mo/ugrid-checks",
]

BASE_DIR = "./"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")
RELEASE_FILE = os.path.join(DATA_DIR, "releases.csv")
IRIS_PULL_CSV = os.path.join(DATA_DIR, "scitools-iris-pulls.csv")
IRIS_ISSUE_CSV = os.path.join(DATA_DIR, "scitools-iris-issues.csv")
IRIS_COMMIT_CSV = os.path.join(DATA_DIR, "scitools-iris-commits.csv")
IRIS_COMMIT_JSON = os.path.join(DATA_DIR, "scitools-iris-commits.json")

# ------------------------------------------------------------------------------


def df_print(df, name, describe=False, dtypes=False):
    if not DEBUG:
        return

    w = os.get_terminal_size().columns

    line = "== df_print: {} ".format(name)
    print(line, end="")
    print("=" * (w - len(line) - 3))

    print(df)
    if describe:
        print("== describe.....")
        print(df.describe())

    if dtypes:
        print("== type.....")
        print(df.dtypes)

    line = "== df_print END"
    print(line, end="")
    print("=" * (w - len(line) - 3))


# ------------------------------------------------------------------------------
# left align a df using formatting
# ------------------------------------------------------------------------------


def left_align(df):
    left_aligned_df = df.style.set_properties(**{"text-align": "left"})
    left_aligned_df = left_aligned_df.set_table_styles(
        [
            dict(selector="th", props=[("text-align", "left")]),
            dict(selector="tr", props=[("text-align", "left")]),
        ]
    )

    return left_aligned_df


start_time = time.time()

gd.autolog_info(f"Starting....")

# ------------------------------------------------------------------------------
# Retrieve the data from Github.
# ------------------------------------------------------------------------------

if QUERY_GITHUB:
    start_time_query = time.time()

    full_path_token = os.path.join(os.getenv("HOME"), ".api_keys", TOKEN_FILE)
    gd.autolog_info(f"Loading GitHub API token from: {full_path_token}")
    token = gd.get_token(full_path_token)

    # https://pygithub.readthedocs.io/en/latest/github.html?highlight=page
    g = Github(token, per_page=100)

    gd.get_releases(g, REPOS)
    gd.get_pulls(g, "scitools/iris")
    gd.get_issues(g, "scitools/iris")
    gd.get_commits_direct("scitools/iris", token)

    gd.autolog_info(
        "--- All Stats queries run duration: {} seconds ---".format(
            round((time.time() - start_time_query), 2)
        )
    )

# ------------------------------------------------------------------------------
# Get all the data we need into pandas
# ------------------------------------------------------------------------------

# ------ pulls --------
gd.autolog_info(f"Pandas: Reading {IRIS_PULL_CSV}")
df_pulls = pd.read_csv(
    IRIS_PULL_CSV,
    sep=",",
    header=0,
    skipinitialspace=True,
    parse_dates=["created_at"],
)

# ------ issues --------
gd.autolog_info(f"Pandas: Reading {IRIS_ISSUE_CSV}")
df_issues = pd.read_csv(
    IRIS_ISSUE_CSV,
    sep=",",
    header=0,
    skipinitialspace=True,
    parse_dates=["created_at", "closed_at"],
)

# ------ commits -------
gd.autolog_info(f"Pandas: Reading {IRIS_COMMIT_CSV}")
df_commits = pd.read_csv(
    IRIS_COMMIT_CSV,
    sep=",",
    header=0,
    skipinitialspace=True,
)

df_commits["week_dt"] = pd.to_datetime(df_commits["week"], unit="s")
df_commits["year"] = pd.to_datetime(df_commits["week"], unit="s").dt.year

# ------ releases ------
gd.autolog_info(f"Pandas: Reading {RELEASE_FILE}")
df_releases = pd.read_csv(
    RELEASE_FILE,
    sep=",",
    header=0,
    skipinitialspace=True,
    keep_default_na=False,
    parse_dates=["published_at"],
)

# Create a clickable url column.  If the url is empty then do not create a link.
def check_empty_url(url, text):
    if url == "":
        return f"{text}"
    else:
        return f"<a href='{url}'>{text}</a>"


df_releases["repo_click"] = df_releases.apply(
    lambda x: check_empty_url(x["repo_url"], x["repo_name"]), axis=1
)

df_releases["release_click"] = df_releases.apply(
    lambda x: check_empty_url(x["release_url"], x["release_name"]), axis=1
)

df_print(df_releases, "df_releases", dtypes=True)
df_releases = df_releases.replace(np.nan, "", regex=True)
df_releases = df_releases.sort_values(by=["published_at"], ascending=False)

df_releases["published_date"] = pd.to_datetime(df_releases["published_at"]).dt.date
df_releases["published_year"] = pd.to_datetime(df_releases["published_at"]).dt.year

# the latest releases for each repo
# Useful: https://stackoverflow.com/questions/41525911/group-by-pandas-dataframe-and-select-latest-in-each-group

df_releases_latest_by_repo = (
    df_releases.sort_values("published_date")
    .groupby("repo_name", as_index=False)
    .apply(lambda x: x.tail(1))
    .reset_index(drop=True)
)
df_releases_latest_by_repo = df_releases_latest_by_repo.sort_values(
    "published_date", ascending=False
)
df_print(
    df_releases_latest_by_repo.reset_index(drop=True),
    "df_releases_latest_by_repo.reset_index()",
)

# ------------------------------------------------------------------------------
# define all stats boxes
# ------------------------------------------------------------------------------

current_year = datetime.date.today().year

releases_this_year = df_releases[df_releases["published_year"] == current_year].shape[0]
releases_last_year = df_releases[
    df_releases["published_year"] == current_year - 1
].shape[0]
releases_last_last_year = df_releases[
    df_releases["published_year"] == current_year - 2
].shape[0]

df_issues_open = df_issues[df_issues["issue_state"] == "open"]
issues_open_count = df_issues_open.shape[0]
pulls_open_count = df_pulls[df_pulls["pr_state"] == "open"].shape[0]
pulls_closed_count = df_pulls[df_pulls["pr_state"] == "closed"].shape[0]
pulls_merged_count = df_pulls[df_pulls["pr_state"] == "closed"].shape[0]

# ------------------------------------------------------------------------------
# Define all plots
# ------------------------------------------------------------------------------
# useful : https://towardsdatascience.com/time-series-and-logistic-regression-with-plotly-and-pandas-8b368e76b19f
gd.autolog_info(f"Creating plots")

# --- SciTools: Release Plot of count
df_releases2 = df_releases.copy()
df_releases2.rename(columns={"published_at": "release_count"}, inplace=True)
df_releases2 = (
    df_releases2.groupby(["published_year", "repo_name"]).size().unstack(level=1)
)

df_print(df_releases2, "df_releases2")

# https://plotly.com/python/bar-charts/
fig_releases_count = px.bar(
    df_releases2,
    barmode="group",
    labels=dict(published_year="Published Year", value="Count"),
)

# add range selector: https://plotly.com/python/range-slider/
fig_releases_count.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=False),
        fixedrange=True,
        type="date",
        tickformat="%Y",
    ),
    yaxis=dict(
        fixedrange=True,
    ),
)

# --- SciTools: Release Plot of total by year
# https://plotly.com/python/bar-charts/
df_releases4 = (
    df_releases.groupby(["published_year", "repo_name"]).size().unstack(level=1)
)
df_print(df_releases4, "df_releases4")

fig_releases_total_by_year = px.bar(
    df_releases4,
    labels=dict(published_year="Published Year", value="Count"),
)

# --- SciTools: Release Plot of total per project
# https://plotly.com/python/bar-charts/
df_releases3 = (
    df_releases.groupby(["repo_name", "published_year"]).size().unstack(level=1)
)
df_print(df_releases3, "df_releases3")

fig_releases_total = px.bar(
    df_releases3,
    labels=dict(repo_name="Repo Name", value="Count"),
)

# --- SciTools: Release timeline
# https://plotly.com/python/line-and-scatter/
# https://plotly.com/python/text-and-annotations/
fig_release_timeline = px.scatter(
    df_releases,
    x="published_at",
    y="repo_name",
    color="repo_name",
    hover_name="release_name",
    labels=dict(published_at="Published Year", repo="Repository"),
)
fig_release_timeline.update_traces(textposition="top center")

# add range selector: https://plotly.com/python/range-slider/
fig_release_timeline.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=False),
        type="date",
    ),
    # margin=dict(t=25),
)

fig_release_timeline.update_traces(
    marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")),
    marker_symbol="circle",
)

# --- Iris: Plots of issues cumulitative sum
df_print(df_issues, "df_issues", dtypes=True)

# add a column with the same time of 00:00:00.  This will allow us to group by date
df_issues["date"] = df_issues.apply(
    lambda row: datetime.datetime.combine(row.created_at, datetime.datetime.min.time()),
    axis=1,
)

df_issues_sum = df_issues.groupby(["date"]).count().cumsum()
df_print(df_issues_sum, "df_issues_sum")

df_issues_closed = df_issues[df_issues["issue_state"] == "closed"]
issues_closed_count = df_issues_closed.shape[0]
df_issues_closed_sum = df_issues_closed.groupby(["date"]).count().cumsum()
df_print(df_issues_closed_sum, "df_issues_closed_sum", dtypes=True)

fig_issues = go.Figure()

fig_issues.add_trace(
    go.Scatter(
        x=df_issues_sum.reset_index()["date"],
        y=df_issues_sum.reset_index()["number"],
        name="Created",
        mode="lines",
        line=dict(color="red"),
    )
)

fig_issues.add_trace(
    go.Scatter(
        x=df_issues_closed_sum.reset_index()["date"],
        y=df_issues_closed_sum.reset_index()["number"],
        name="Closed",
        mode="lines",
        line=dict(color="green"),
    )
)

fig_issues.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=0, b=25, r=5, l=5),
)

fig_issues.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=False),
        type="date",
    )
)

# --- Iris: Plots of issues daily
df_print(df_issues, "df_issues", dtypes=True)

df_issues_daily = df_issues.groupby(["date"]).count()
issues_daily_mean = df_issues_daily["title"].mean()
issues_daily_max = df_issues_daily["title"].max()

fig_issues_daily = go.Figure()

fig_issues_daily.add_trace(
    go.Bar(
        x=df_issues_daily.reset_index()["date"],
        y=df_issues_daily.reset_index()["number"],
        marker_color=["red"] * df_issues_daily.shape[0],
    )
)
fig_issues_daily.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=0, b=25, r=5, l=5),
    bargap=0.1,
)

fig_issues_daily.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=False),
        type="date",
    )
)

# --- Iris: Plots of pull requests
df_print(df_pulls, "df_pulls", dtypes=True)

df_pulls_open = df_pulls[df_pulls["pr_state"] == "open"]

# add a column with the same time of 00:00:00.  This will allow us to group by date
df_pulls["date"] = df_pulls.apply(
    lambda row: datetime.datetime.combine(row.created_at, datetime.datetime.min.time()),
    axis=1,
)

df_pulls_sum = df_pulls.groupby(["date"]).count().cumsum()
df_print(df_pulls_sum, "df_pulls_sum")

df_pulls_closed = df_pulls[df_pulls["pr_state"] == "closed"]
df_pulls_closed_sum = df_pulls_closed.groupby(["date"]).count().cumsum()

fig_pulls = go.Figure()

fig_pulls.add_trace(
    go.Scatter(
        x=df_pulls_sum.reset_index()["date"],
        y=df_pulls_sum.reset_index()["number"],
        name="Created",
        mode="lines",
        line=dict(color="red"),
    )
)

fig_pulls.add_trace(
    go.Scatter(
        x=df_pulls_closed_sum.reset_index()["date"],
        y=df_pulls_closed_sum.reset_index()["number"],
        name="Closed",
        mode="lines",
        line=dict(color="green"),
    )
)

fig_pulls.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10),
)
# add range selector: https://plotly.com/python/range-slider/
fig_pulls.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=False),
        type="date",
    )
)

# --- Iris: Plots of contributors
df_print(df_commits, "df_commits", dtypes=True)

df_commits_by_author = df_commits
df_commits_by_author = df_commits_by_author.groupby(["author"]).sum()
df_commits_by_author = df_commits_by_author[df_commits_by_author["commits"] > 5]

df_print(df_commits, "df_commits_by_author", dtypes=True)

fig_commits = go.Figure()

fig_commits.add_trace(
    go.Bar(
        x=df_commits_by_author.reset_index()["author"],
        y=df_commits_by_author.reset_index()["commits"],
        # name="Created",
        # line=dict(color="blue"),
    )
)

fig_commits.update_layout(hovermode="x unified", margin=dict(t=0))

# --- Iris: Contributors info
contributors_list = df_commits["author"].unique().tolist()

if QUERY_GITHUB:
    gd.autolog_info(f"GitHub: Retrieving {len(contributors_list)} avatars images")
    gd.check_dir(IMAGE_DIR)

    for author in contributors_list:
        avatar_url = (
            df_commits.query('author == "' + author + '"')
            .head(1)["avatar_url"]
            .values[0]
        )
        response = requests.get(avatar_url)

        image_file = os.path.join(IMAGE_DIR, author + ".png")
        with open(image_file, "wb") as f:
            f.write(response.content)

contributor_images = [
    dp.Group(dp.Media(file=f"images/{author}.png"), dp.Text(author), columns=1)
    for author in contributors_list
]

# --- Iris: Plots of contributions by week
df_print(df_commits, "df_commits", dtypes=True)

df_commits_weekly = df_commits
df_commits_weekly = df_commits_weekly.groupby(["week_dt"]).sum()

df_print(df_commits_weekly, "df_commits_weekly", dtypes=True)

fig_commits_weekly = go.Figure()

fig_commits_weekly.add_trace(
    go.Bar(
        x=df_commits_weekly.reset_index()["week_dt"],
        y=df_commits_weekly.reset_index()["commits"],
        name="Commits",
        # line=dict(color="blue"),
    )
)

fig_commits_weekly.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=0),
)

# -- Iris: Plots of new contributors by year
df_print(df_commits, "df_commits")

df_contributors_by_year = df_commits
# strip out all the non commit weeks
df_contributors_by_year = df_contributors_by_year[
    df_contributors_by_year["commits"] > 0
]
df_contributors_by_year = df_contributors_by_year.sort_values(
    by=["week", "author"], ascending=True
)
df_print(df_contributors_by_year, "df_contributors_by_year ---1---")

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
df_contributors_by_year = df_contributors_by_year.drop_duplicates(
    "author", keep="first", inplace=False
)
df_print(df_contributors_by_year, "df_contributors_by_year ---2---")
df_contributors_by_year2 = df_contributors_by_year.groupby(["year"]).count()
df_print(df_contributors_by_year, "df_contributors_by_year ---3---")

# -- Iris: Contributors stat boxes

commits_user_count = df_commits["commits"].sum()
contributors_count = df_contributors_by_year.shape[0]
contributors_count_this_cy = df_contributors_by_year[
    df_contributors_by_year["year"] == 2021
].shape[0]
contributors_count_last_cy = df_contributors_by_year[
    df_contributors_by_year["year"] == 2020
].shape[0]

fig_contributors_by_year = go.Figure()
fig_contributors_by_year.add_trace(
    go.Scatter(
        x=df_contributors_by_year2.reset_index()["year"],
        y=df_contributors_by_year2.reset_index()["commits"],
        name="Commits",
        line=dict(color="blue"),
    )
)
fig_contributors_by_year.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=0),
)

# rename some columns so the output is easier to read
RELEASE_TABLE_COLUMNS = ["Repository", "Release", "Published"]
RELEASE_TABLE_COLUMNS_CLICK = ["Repository (link)", "Release (link)", "Published"]

df_releases.rename(
    columns={
        "repo_name": "Repository",
        "release_name": "Release",
        "repo_click": "Repository (link)",
        "release_click": "Release (link)",
        "published_date": "Published",
    },
    inplace=True,
)

df_releases_latest_by_repo.rename(
    columns={
        "repo_click": "Repository (link)",
        "release_click": "Release (link)",
        "published_date": "Published",
    },
    inplace=True,
)

# ------------------------------------------------------------------------------
# Create the report
# ------------------------------------------------------------------------------

gd.autolog_info(f"Creating DataPane Report")

now = datetime.datetime.now()

md_about = """
## Challenge 2020+

The 2020 Challenge aims to double the frequency of deployments to production
without breaking quality.  For more information see the
[AVD Challenge 2020+](https://metoffice.sharepoint.com/sites/TechSolDelAVDTeam/SitePages/AVD-Challenge-2020.aspx)
on our SharePoint site.

The AVD Team has taken the original Challenge 2020 and taken it further to
show all past releases for a variety of projects including some more specific
analysis of the Iris Project (separate tab), we have dubbed this the
*Challenge 2020+*.


## Tips for Interacting with Plots and Tables

* Plots
   * Click on an item in the legend to show/hide it
   * Select an area in a plot to zoom in
   * Double click in a plot to reset the zoom
   * Select the period (if shown) in the top left corner,
     typically 1y, 2y, 3y or all (years)
* Tables
   * Any tables of data can be sorted by clicking on the column heading
   * Filtters can be applied to tables, mouse over a column heading and
     select the filter icon to the right
* Refresh the page to reset all interactions

## Could the data shown be improved or more added?

Yes.  This can be an evolving report that could be extended to include
more plots and analysis.  It could also be reused for other GitHub based
projects with little effort.


## How was this page created?

The intent for this page to be self contained with no dependencies including
not having a web service  to be online apart from a shared httpd service
to serve the html.

After some research I chose to experiment and use
[datapane](https://datapane.com/).  This tools is still evolving but already
allows for static reports (thie reports is a single html file) to be created
with little knowlege outside of Python, and is of course free to use.

A python program was used to create the report, the high level approach is:

  1. Use the GitHub API via [PyGitHub](PyGithub) to retrieve stats
     1. A list of REPOS to query are in the source.
     1. For each of the REPOS the API is used to fetch data for:
        1. Releases
        1. Pull requests
        1. Issues
        1. Commits
  1. The API query returns a json file that is then filtered and a
     corresponding csv file is created.
  1. [pandas](https://pandas.pydata.org/pandas-docs/stable/) is then used the
     load the data.  pandas was chosen as it allows easy maniulation and
     ultimately plotting of tabular data.
  1. [plotly](https://plotly.com/python/getting-started/) is then used to
     create the figures
  1. [datapane](https://datapane.com/) is then used to create a report by
     using the plotly figures and pandas dataframes.


## Is this version controlled?

Yes.  See https://bitbucket.org/metoffice/codeshares/src/master/challenge2020plus/

**Disclaimer**: This is not a production used report.  It is used for information
and learning purposed only.
"""

md_intro = """
You can peruse all GitHub releases directly on their website.  For example
the Iris releases can be found here: https://github.com/SciTools/iris/releases.
For more information on the data sources used see the
[README](https://bitbucket.org/metoffice/codeshares/src/master/challenge2020plus/README.md).
"""

text_created = "<b>Report Generated: " + now.strftime("%d-%m-%Y %H:%M:%S") + "</b>"

report = dp.Blocks(
    dp.Page(
        dp.Text(f"## SciTools & Scientific Software Stack"),
        dp.Text(md_intro),
        dp.Text(text_created),
        dp.Group(
            dp.BigNumber(
                heading="Releases Last Last CY", value=releases_last_last_year
            ),
            dp.BigNumber(
                heading="Releases Last CY",
                value=releases_last_year,
                change=abs(releases_last_year - releases_last_last_year),
                is_upward_change=False
                if releases_this_year < releases_last_year
                else True,
            ),
            dp.BigNumber(
                heading="Releases This CY",
                value=releases_this_year,
                change=abs(releases_this_year - releases_last_year),
                is_upward_change=False
                if releases_this_year < releases_last_year
                else True,
            ),
            columns=3,
        ),
        dp.Plot(
            fig_release_timeline,
            caption=f"Release Timeline",
        ),
        dp.Plot(
            fig_releases_total_by_year,
            caption=f"Release total count by year",
        ),
        dp.Plot(
            fig_releases_count,
            caption=f"Release count by year",
            label="Plot",
        ),
        dp.Plot(fig_releases_total, caption=f"Release since project creation"),
        dp.Group(
            dp.Table(
                left_align(
                    df_releases_latest_by_repo[RELEASE_TABLE_COLUMNS_CLICK].head(30)
                ),
                caption=f"Releases by most recent per repo",
            ),
            columns=1,
        ),
        dp.Select(
            dp.DataTable(
                df_releases[RELEASE_TABLE_COLUMNS],
                caption=f"Release History",
                label="Table (interactive)",
            ),
            dp.Table(
                left_align(df_releases[RELEASE_TABLE_COLUMNS_CLICK].head(30)),
                caption=f"Release History",
                label="Table (non interactive, clickable links, 30 most recent)",
            ),
        ),
        title="Challenge 2020+",
    ),
    # --------------------------------------------------------------------------
    dp.Page(
        dp.Text(f"## Issues"),
        dp.Group(
            dp.BigNumber(heading="Open Issues", value=issues_open_count),
            dp.BigNumber(heading="Closed Issues", value=issues_closed_count),
            dp.BigNumber(
                heading="Average Daily Issues", value=round(issues_daily_mean, 1)
            ),
            dp.BigNumber(heading="Max Daily Issues", value=issues_daily_max),
            columns=4,
        ),
        dp.Group(
            dp.Plot(fig_issues, caption=f"Iris Issues over Time"),
            dp.Plot(fig_issues_daily, caption=f"Iris Issues by Day"),
            columns=2,
        ),
        dp.Select(
            dp.DataTable(
                df_pulls_open[
                    ["number", "pr_state", "title", "user", "created_at", "updated_at"]
                ],
                caption=f"Open Pull Requests",
                label="Open Pull Requests",
            ),
            dp.DataTable(df_issues_open, caption=f"Open Issues", label="Open Issues"),
        ),
        dp.Text(f"## Pull Requests"),
        dp.Group(
            dp.BigNumber(heading="Open Pull Requests", value=pulls_open_count),
            dp.BigNumber(heading="Closed Pull Requests", value=pulls_closed_count),
            columns=4,
        ),
        dp.Group(
            dp.Plot(fig_pulls, caption=f"Iris Pull Requests"),
            columns=1,
        ),
        dp.Text(f"## Contributors"),
        dp.Group(
            dp.BigNumber(heading="Contributors", value=contributors_count),
            dp.BigNumber(
                heading="New Contributors last CY", value=contributors_count_last_cy
            ),
            dp.BigNumber(
                heading="New Contributors this CY",
                value=contributors_count_this_cy,
                change=abs(contributors_count_this_cy - contributors_count_last_cy),
                is_upward_change=False
                if contributors_count_this_cy < contributors_count_last_cy
                else True,
            ),
            dp.BigNumber(heading="Total User Commits", value=commits_user_count),
            columns=4,
        ),
        dp.Plot(fig_commits_weekly, caption=f"Iris Contributions by week"),
        dp.Plot(fig_commits, caption=f"Iris Contributors"),
        dp.Plot(fig_contributors_by_year, caption=f"New Iris Contributors by Year"),
        dp.Text(f"### All Contributors"),
        dp.Group(
            *contributor_images,
            columns=7,
        ),
        title="Iris Stats",
    ),
    dp.Page(dp.Text(md_about), title="About"),
)

gd.autolog_info(f"Saving DataPane Report: {REPORT_HTML}")
dp.save_report(
    report,
    path=REPORT_HTML,
    open=False,
    # https://docs.datapane.com/reports/configuring-reports/styling
    formatting=dp.Formatting(
        light_prose=False,
        accent_color="blue",
        bg_color="#EEE",
    ),
)

gd.autolog_info(
    "{}, Total run duration: {} seconds ---".format(
        os.path.basename(__file__), round((time.time() - start_time), 2)
    )
)
