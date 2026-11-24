"""
This is code to test a submission for the IEEE SSCI 2027 MAD Chairs Tournament.

MAD Chairs is a game repeated for multiple rounds. In each round, each player
selects from a set of resources (e.g. "A", "B", "C", "D" or "E") or selects to
"skip". Each player who selects a resource no other player selects for that
round wins that round.

In this test, MAD Chairs matches are simulated which involve three players of
one strategy vs three players of another strategy. Each player name is
composed of its strategy name followed by a letter. Your strategy name will be
"challenger" so the three players of your strategy will be "challengerA",
"challengerB", and "challengerC".

Before running this test in https://colab.research.google.com/, you would first
need to run the following in a separate cell:

    !pip install trueskill

Set strategy_name to the name of your MAD Chairs strategy and code it in the
submission() function.

The results will output in three files:
- "{strategy_name}_results.csv" shows what the players selected in each round of
    each match and how often they won (as a %)
- "{strategy_name}_stats.csv" shows how well each strategy performed against the
    other strategies
- "{strategy_name}_submission.csv" is the file to submit on Kaggle. It
    contains only the skill score of your submission.
"""
import pandas as pd
import random
import math
import trueskill

strategy_name = "My_strategy" # <-- set this to your strategy name

def submission(
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict = {}
) -> str:
  """
  TODO: Enter documentation about your submission here.
  """
  return "TODO: Code me"

# Constants
schedule = pd.read_csv("/kaggle/input/datasets/chrissantoslang/schedule/schedule.csv")
rounds = 20
resources = ["A", "B", "C", "D", "E"]
num_players = 6
strategies = ["challenger", "turntaking", "equalize", "caste", "radicaleq",
  "rotate0", "random3", "rotate", "random"]
rationality = 50

def randomN(
    n: int,
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict
) -> str:
  """
  randomN() returns a random resource in round 1. Afterwards, it always repeats
  its previous selection when it won but only (n-1)/n of the time when it lost.
  If changing, it randomly selects from the resources selected least in the
  previous round.
  """
  if round == 1:
    return random.choice(resources)
  previous = history[f"Round{round-1}"].tolist()
  if (
      (random.random() < 1/n) and
      cache["wins"].loc[position - 1, f"Round{round-1}"] == 0
  ):
    options = [b for b in resources if b not in previous]
    if len(options) < 1:
      options = [b for b in resources if previous.count(b) == 1]
    if len(options) < 1:
      options = resources
    return random.choice(options)
  return previous[position - 1]

def radicaleq(
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict
) -> str:
  """
  radicaleq returns the top resource for the player who has won the least
  (i.e. the least wealthy), then the next resource for the next wealthiest,
  etc. After each resource has been assigned, all remaining players are
  assigned to 'skip'. In the case of ties, the player with later position
  counts as wealthier.
  """
  if cache.get("reqRound") != round:
    cache["reqRound"] = round
    updatePopularity(round, cache, history)
    if round == 1:
      cache["radicaleq"] = cover
    else:
      sorted_res = [r[0] for r in sorted(cache["popularity"].items(), key=lambda item: item[1])]
      sorted_pos = [pos for bonus, pos in sorted([(cache["wins"]["total"].tolist()[p], p) for p in range(num_players)])]
      cache["radicaleq"] = [s[1] for s in sorted(zip(sorted_pos, (sorted_res + extra)))]
  return cache["radicaleq"][position - 1]

def equalize(
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict
) -> str:
  """
  equalize() is like radicaleq(), but ingroup/outgroup accounting is added:
  Maintain a count of deviations from the strategy discounted by 30%
  per round (so deviations in the distant past will be forgiven). Any
  player who has deviated at least once (after discounting) is in the outgroup
  and counted as wealthier than everyone in the ingroup.
  """
  if cache.get("eqRound") != round:
    cache["eqRound"] = round
    updatePopularity(round, cache, history)
    if round == 1:
      cache["eqViolations"] = [0] * num_players
      cache["equalize"] = cover
    else:
      cache["eqViolations"] = [v * 0.7 for v in cache["eqViolations"]]
      for player in history.itertuples():
        choice = getattr(player, f"Round{round - 1}")
        if choice != cache["equalize"][player.Position - 1]:
          cache["eqViolations"][player.Position - 1] += 1
      sorted_res = [r[0] for r in sorted(cache["popularity"].items(), key=lambda item: item[1])]
      sorted_pos = [pos for bonus, pos in sorted([(C["wins"]["total"].tolist()[p], p) for p in range(num_players)])]
      learners = [p for p in sorted_pos if cache["eqViolations"][p] < 1]
      sorted_pos = learners + [p for p in sorted_pos if p not in learners]
      cache["equalize"] = [s[1] for s in sorted(zip(sorted_pos, (sorted_res + extra)))]
  return cache["equalize"][position - 1]

def caste(
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict
) -> str:
  """
  caste() uses the Trueskill algorithm to maintain skill-estimates for all
  players, uses those estimates to predict probabilities of winning, and
  maintains accounts of favors owed between all players, where debt incurred
  from beating a player is the probability of that other player
  winning and debt incurred from tying is that same probability minus
  ones own probability of winning. Returns as with equalize(), substituting
  credit for wealth.
  """
  if cache.get("casteRound") != round:
    cache["casteRound"] = round
    updatePopularity(round, cache, history)
    trackFavors(round, cache, history)
    if round == 1:
      cache["casteViolations"] = [0] * num_players
      cache["caste"] = cover
    else:
      cache["casteViolations"] = [v * 0.7 for v in cache["casteViolations"]]
      for player in history.itertuples():
        choice = getattr(player, f"Round{round - 1}")
        if choice != cache["caste"][player.Position - 1]:
          cache["casteViolations"][player.Position - 1] += 1
      sorted_res = [r[0] for r in sorted(cache["popularity"].items(), key=lambda item: item[1])]
      sorted_pos = [pos for debt, pos in sorted([(-cache["debt"][p], p) for p in range(num_players)])]
      learners = [p for p in sorted_pos if cache["casteViolations"][p] < 1]
      sorted_pos = learners + [p for p in sorted_pos if p not in learners]
      cache["caste"] = [s[1] for s in sorted(zip(sorted_pos, (sorted_res + extra)))]
  return cache["caste"][position - 1]

def turntaking(
    position: int,
    round: int,
    history: pd.DataFrame,
    cache: dict
) -> str:
  """
  turntaking() returns as with caste(), substituting debt for credit.
  """
  if cache.get("turnRound") != round:
    cache["turnRound"] = round
    updatePopularity(round, cache, history)
    trackFavors(round, cache, history)
    if round == 1:
      cache["turnViolations"] = [0] * num_players
      cache["turntaking"] = extra + resources
    else:
      cache["turnViolations"] = [v * 0.6 for v in cache["turnViolations"]]
      for player in history.itertuples():
        choice = getattr(player, f"Round{round - 1}")
        if choice != cache["turntaking"][player.Position - 1]:
          cache["turnViolations"][player.Position - 1] += 1
      sorted_res = sorted(cache["popularity"].items(), key=lambda item: item[1])
      sorted_pos = [pos for debt, pos in sorted([(-cache["debt"][p], p) for p in range(num_players)])]
      learners = [p for p in sorted_pos if cache["turnViolations"][p] < 1]
      turnTakers = [p for p in sorted_pos if p not in learners] + learners
      free = [r for r in sorted_res if r[1] < round * 1.45]
      sorted_res = [r[0] for r in [x for x in sorted_res if x not in free] + free]
      cache["turntaking"] = [s[1] for s in sorted(zip(turnTakers, (extra + sorted_res)))]
  return cache["turntaking"][position - 1]

def updatePopularity(round: int, cache: dict, hist: pd.DataFrame):
  """
  Used by turntaking(), caste(), equalize(), and radicaleq() to order the
  resources by their historic popularity. The results are cached in
  cache["popularity"].
  """
  if cache.get("popRound") != round:
    cache["popRound"] = round
    if round == 1:
      cache["popularity"] = {name: 0 for name in resources}
    else:
      for player in hist.itertuples():
        choice = getattr(player, f"Round{round - 1}")
        if choice in resources:
          cache["popularity"][choice] += 1 + (player.Position)/10000
  return

def trackFavors(round: int, cache: dict, hist: pd.DataFrame):
  """
  Used by turntaking() and caste() to track favors between players. The results
  are cached in cache["debt"].
  """
  if cache.get("skillRound") == round:
    return
  cache["skillRound"] = round
  if round == 1:
    cache["skill"] = [trueskill.Rating(mu=25, sigma=0.5)] * num_players
    cache["debt"] = [(num_players - p + 1)/100000 for p in range(num_players)]
  else:
    selections = hist[f"Round{round - 1}"].tolist()
    winners = []
    for pos, selection in enumerate(selections):
      if selection != "skip" and selections.count(selection) == 1:
        winners.append(0)
      else:
        winners.append(1)
    for me in range(num_players):
      myRating = cache["skill"][me]
      for other in range(num_players):
        if me != other:
           otherRating = cache["skill"][other]
           denom = math.sqrt(2 * trueskill.BETA**2 + myRating.sigma**2 + otherRating.sigma**2)
           if myRating.mu == otherRating.mu:
             myWin = 0.5
           elif myRating.mu > otherRating.mu:
             myWin = trueskill.global_env().cdf((myRating.mu - otherRating.mu) / denom)
           else:
             myWin = 1 - trueskill.global_env().cdf((otherRating.mu - myRating.mu) / denom)
           nondraw = 1 - trueskill.quality_1vs1(myRating, otherRating)
           if (winners[me] == 1) or (winners[other] == 0):
             cache["debt"][me] -= myWin * nondraw
           if (winners[me] == 0) or (winners[other] == 1):
             cache["debt"][me] += (1 - myWin) * nondraw
    cache["skill"] = [s[0] for s in trueskill.rate([(r,) for r in cache["skill"]], winners)]
  return

def initStats():
  """
  Prepare the stats dataframe.
  """
  stats = pd.DataFrame({"Strategy": strategies}, index=strategies)
  for stat in ["matches", "wins", "edge"]:
    for strat in strategies[::-1]:
      stats[f"{stat}_v_{strat}"] = 0.0
  return stats

def getSelection(
    strategy: str,
    pos: int,
    round: int,
    hist: pd.DataFrame,
    cache: dict={},
    champion_cache: dict={}
) -> str:
  if strategy.upper() == "ROTATE":
    return cover[((pos + round - 2) % num_players)]
  elif strategy.upper() == "ROTATE0":
    return cover[pos - 1]
  elif strategy.upper() == "RANDOM":
    return random.choice(resources + ["skip"])
  elif strategy.upper() == "RANDOM3":
    return randomN(3, pos, round, hist, cache)
  elif strategy.upper() == "RADICALEQ":
    return radicaleq(pos, round, hist, cache)
  elif strategy.upper() == "EQUALIZE":
    return equalize(pos, round, hist, cache)
  elif strategy.upper() == "CASTE":
    return caste(pos, round, hist, cache)
  elif strategy.upper() == "TURNTAKING":
    return turntaking(pos, round, hist, cache)
  return submission(pos, round, hist, champion_cache)

#Process the file
extra = ["skip"] * (num_players - len(resources))
cover = resources + extra
stats = initStats()
results = pd.DataFrame()
ratings = {strat: trueskill.Rating(mu=25.000, sigma=8.333) for strat in strategies}
for match in schedule.itertuples():

  #Initialize match
  names = match[2:]
  matchResults = pd.DataFrame({
    "Match": [match.Match] * num_players,
    "Player": names,
    "Strategy": [p[:-1] for p in names],
    "Position": range(1, num_players + 1),
  })
  strat1 = matchResults["Strategy"].iloc[0]
  if matchResults["Strategy"].nunique() == 1:
    strat2 = strat1
  else:
    strat2 = matchResults.loc[~matchResults["Strategy"].isin([strat1]), "Strategy"].iloc[0]
  C = {"wins": pd.DataFrame({"total": [0] * num_players})}
  champion_cache = {}

  #Play rounds
  for round in range(1, rounds + 1):
    colName = f"Round{round}"
    new_column = []
    for player in matchResults.itertuples():
      new_column.append(getSelection(
        player.Strategy,
        player.Position,
        round,
        matchResults.iloc[:, 3:],
        C,
        champion_cache
      ))
    matchResults = pd.concat([matchResults, pd.Series(new_column, name=colName)], axis=1)

    #update score
    new_column = []
    for player in matchResults.itertuples():
      selection = getattr(player, colName)
      if selection != "skip" and (matchResults[colName] == selection).sum() == 1:
        new_column.append(1)
      else:
        new_column.append(0)
    C["wins"] = pd.concat([C["wins"], pd.Series(new_column, name=colName)], axis=1)
    C["wins"]["total"] = C["wins"]["total"] + C["wins"][colName]
    #end of round

  matchResults["Win"] = C["wins"]["total"]/rounds

  #update stats
  stats.at[strat1,f"matches_v_{strat2}"] += 1
  if strat1 != strat2:
    stats.at[strat2,f"matches_v_{strat1}"] += 1
  strat1_min = 2
  strat2_min = 2
  strat1_wins = 0
  strat2_wins = 0
  for player in matchResults.itertuples():
    if strat1 == strat2:
      stats.at[strat1,f"wins_v_{strat1}"] += player.Win / num_players
      strat1_wins += player.Win / num_players
    elif player.Strategy == strat1:
      stats.at[strat1,f"wins_v_{strat2}"] += player.Win * 2 / num_players
      strat1_min = min(strat1_min, player.Win)
      strat1_wins += player.Win * 2 / num_players
    elif player.Strategy == strat2:
      stats.at[strat2,f"wins_v_{strat1}"] += player.Win * 2 / num_players
      strat2_min = min(strat2_min, player.Win)
      strat2_wins += player.Win * 2 / num_players
  (edge1, edge2) = (0, 0)    
  if strat1 != strat2:
    edge1 = max(-1, min(1, (strat1_min - strat2_min)/max(0.001, strat2_min)))
    edge2 = max(-1, min(1, (strat2_min - strat1_min)/max(0.001, strat1_min)))
    stats.at[strat1,f"edge_v_{strat2}"] += edge1
    stats.at[strat2,f"edge_v_{strat1}"] += edge2

  strat1_ewins = strat1_wins * (rationality ** edge1)
  strat2_ewins = strat2_wins * (rationality ** edge2)
  if (strat1 == strat2) or (strat1_ewins == strat2_ewins):
    ratings[strat1], ratings[strat2] = trueskill.rate_1vs1(ratings[strat1], ratings[strat2], drawn = True)
    matchResults["Winner"] = "draw"
  elif strat1_ewins > strat2_ewins:
    ratings[strat1], ratings[strat2] = trueskill.rate_1vs1(ratings[strat1], ratings[strat2])
    matchResults["Winner"] = strat1
  else:
    ratings[strat2], ratings[strat1] = trueskill.rate_1vs1(ratings[strat2], ratings[strat1])
    matchResults["Winner"] = strat2

  results = pd.concat([results, matchResults], ignore_index=True)
  #end of match

#generate outputs
for strat in strategies:
  stats[f"wins_v_{strat}"] = (stats[f"wins_v_{strat}"] / stats[f"matches_v_{strat}"]).round(2)
  stats[f"edge_v_{strat}"] = (stats[f"edge_v_{strat}"] / stats[f"matches_v_{strat}"]).round(2)
  stats.at[strat, "wins_v_self"] = stats.at[strat, f"wins_v_{strat}"]

for strat in strategies[::-1]:
  stats[f"ewins_v_{strat}"] = stats.apply(
    lambda row: min(row["wins_v_self"], row[f"wins_v_{strat}"] * (rationality ** row[f"edge_v_{strat}"])),
    axis=1
  ).round(4)

stats["mean_ewins"] = 0.0
for strat in strategies:
  stats["mean_ewins"] += stats[f"ewins_v_{strat}"] / len(strategies)
  stats.at[strat, "skill"] = ratings[strat].mu / ratings["turntaking"].mu

results.to_csv(f"{strategy_name}_results.csv", index=False)
stats.to_csv(f"{strategy_name}_stats.csv")
print(f"{strategy_name} skill = {stats.at['challenger', 'skill']}")
to_submit = pd.DataFrame({"id":[0],"skill":[stats.at['challenger', 'skill']]})
to_submit.to_csv(f"{strategy_name}_submission.csv", index=False)
