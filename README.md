<img src="https://imgur.com/zUKIKT7.png" style="float: center; margin: 15px">

# Problem Statement:

---

March Madness is one of the biggest, most exciting and most fun events in all of sports.  The NCAA Division I men’s basketball tournament is a single-elimination tournament of 68 teams that compete in seven rounds for the national championship.  The first NCAA bracket pool started in 1977 in a Staten Island Bar.  Eighty-eight people filled out brackets in the pool that year and paid ten dollars in a winner take all format.   At the same bar, in 2006, 150,000 entered and prize money exceeded one and a half million dollars.  

Every year, tens of millions of brackets are filled out through major online bracket games.  Every one of those millions of brackets has one goal - to be perfect.  How hard is it to pick a perfect bracket?  2^63 = 9,223,372,036,854,775,808.  That's 9.2 **QUINTILLION!**  The odds are astronomical against picking a perfect bracket.

The goal with this project is to see if we can use Machine Learning to try and create a perfect bracket.

# Overview:

---

As a result of the continued collaboration between Google Cloud and the NCAA®, the eighth annual Kaggle-backed March Madness® competition was announced Fenruary 2021.  Another year, another chance to anticipate the upsets, call the probabilities, and put bracketology skills to the leaderboard test. Kagglers attempt to forecast the outcomes of March Madness during the NCAA Division I Men’s and Women’s Basketball Championships. But unlike most fans, Kagglers will pick their bracket using a combination of NCAA’s historical data and their computing power.

In the first stage of the competition, Kagglers rely on results of past tournaments to build and test models. 

In the second stage, competitors forecast outcomes of all possible matchups in the 2021 NCAA Division I Men’s and Women’s Basketball Championships. Kagglers don't need to participate in the first stage to enter the second. The first stage exists to incentivize model building and provide a means to score predictions. The real competition is the second stage and forecasting the 2021 results.

Google is the official public cloud provider of the NCAA®. As part of its journey to the cloud, the NCAA has migrated 80+ years of historical and play-by-play data, from 90 championships and 24 sports, to Google Cloud Platform (GCP). The NCAA has tapped into decades of historical basketball data using BigQuery, Cloud Spanner, Datalab, Cloud Machine Learning and Cloud Dataflow, to power the analysis of team and player performance. 

# EDA

---

With the colloboration between google and the NCAA® there is a lot prior years data - back to 1985 - that cover everything as simple as the scores of the Championship Game to literally every single event that took place in a particular game during the regular season.

One of the first hurdles was to just look through the volums of data to see what was available.  Google provides 26 different files of data - several of which have over 2,600,000 rows!  It's a lot of data but it is very clean.

The NCAA bases their schedule on the DayZero column.  DayZero tells you the date corresponding to DayNum = 0 during that season. All game dates are aligned upon a common scale so that (each year) 

- DayNum = 132 is Selection Sunday
- DayNum = 132 is also the final day of the regular season
- DayNum = 134/135 are the days they "play-in" games are played
- DayNum = 152 is the day the National Semifinals are always on
- DayNum = 154 is the Monday the Championship Game of the men's tournament

All game data includes the day number in order to make it easier to perform date calculations. If you need to know the exact date a game was played on, you can combine the game's "DayNum" with the season's "DayZero". For instance, since day zero during the 2011-2012 season was 10/31/2011, if we know that the earliest regular season games that year were played on DayNum = 7, they were therefore played on 11/07/2011.

Some of the more interesting things I noticed upon initially exploring the data is that every team is assigned a 4 digit number for their TeamID.  Then there is another file that has just the team names with the TeamID.  So in order to make a lot of the .csv files useful you need to merge several different .csv files into a panda to work with.

The same holds true for the players.  There's a file that has just the player's first and last name with their PlyerID and their TeamID.

The data files I found most interesting were the "MEventsYYYY.csv" files.  Each MEvents file lists the play-by-play event logs for more than 99.5% of games from that season. Each event is assigned to either a team or a single one of the team's players. Thus if a basket is made by one player and an assist is credited to a second player, that would show up as two separate records. The players are listed by PlayerID within the MPlayers.csv file.

In 2020 google and the  NCAA® transitioned to a different play-by-play source, which includes data since the 2014-2015 season rather than since the 2009-2010 season (that's what they had previously provided for men's data). However, now they are able to provide play-by-play for both men's and women's data, and there is locational play-by-play detail starting with games from the 2018-2019 season. This includes an X/Y location (ranging from 0 to 100 in each dimension) on the court for each shot attempt, turnover, and foul for many games, as well as an overall categorization of the area on the court that the shot or turnover or foul occurred in (inside left wing, outside right wing, under the basket, etc.) Some games in these recent seasons still lack the locational detail. The data from the 2019 season matches what you can expect for the current year (2021 season) as we approach the postseason. Despite the 99.5% coverage, there are still a few games missing annually.

- EventID - this is a unique ID for each logged event. The EventID's are different within each year and uniquely identify each play-by-play event. They ought to be listed in chronological order for the events within their game.
- Season, DayNum, WTeamID, LTeamID - these four columns are sufficient to uniquely identify each game. The games are a mix of Regular Season, NCAA® Tourney, and Secondary Tourney games.
- WFinalScore, LFinalScore - these two columns match the WScore and LScore numbers as found elsewhere in the Compact Results and Detailed Results files. They are provided here to indicate the final score at the end of the game. Note that the event-by-event totals are not guaranteed to add up to the final scores, due to possible data recording errors in the play-by-play.
- WCurrentScore, LCurrentScore - whenever a scoring play happens (1 point, 2 points, or 3 points) the updated score is provided, from the perspective of the winning team (WPoints) and the losing team (LPoints), although of course during the game we didn't know yet that they were the winning team or losing team. Note that in the earlier years of the play-by-play data from this source, the running WCurrentScore and LCurrentScore were not calculated, and so they show up as zero throughout the event log for those years. However, they can still be calculated manually when looping through the game events that are present, by watching for rows like "made1", "made2", and "made3" that represent scoring events.
- ElapsedSeconds - this is the number of seconds that have elapsed from the start of the game until the event occurred. With a 20-minute half, that means that an ElapsedSeconds value from 0 to 1200 represents an event in the first half, a value from 1200 to 2400 represents an event in the second half, and a value above 2400 represents an event in overtime. For example, since overtime periods are five minutes long (that's 300 seconds), a value of 2699 would represent one second left in the first overtime.
- EventTeamID - this is the ID of the team that the event is logged for, which will either be the WTeamID or the LTeamID.
- EventPlayerID - this is the ID of the player that the event is logged for, as described in the MPlayers.csv file.
- EventType, EventSubType - these indicate the type of the event that was logged (see listing below).

Event Types and Subtypes:

- assist - an assist was credited on a made shot
- block - a blocked shot was recorded
- steal - a steal was recorded
- sub - a substitution occurred, with one of the following subtypes: in=player entered the game; out=player exited the game; start=player started the game
- timeout - a timeout was called, with one of the following subtypes: unk=unknown type of timeout; comm=commercial timeout; full=full timeout; short= short timeout
- turnover - a turnover was recorded, with one of the following subtypes: unk=unknown type of turnover; 10sec=10 second violation; 3sec=3 second violation; 5sec=5 second violation; bpass=bad pass turnover; dribb=dribbling turnover; lanev=lane violation; lostb=lost ball; offen=offensive turnover (?); offgt=offensive goaltending; other=other type of turnover; shotc=shot clock violation; trav=travelling
- foul - a foul was committed, with one of the following subtypes: unk=unknown type of foul; admT=administrative technical; benT=bench technical; coaT=coach technical; off=offensive foul; pers=personal foul; tech=technical foul
- fouled - a player was fouled
- reb - a rebound was recorded, with one of the following subtypes: deadb=a deadball rebound; def=a defensive rebound; defdb=a defensive deadball rebound; off=an offensive rebound; offdb=an offensive deadball rebound
- made1, miss1 - a one-point free throw was made or missed, with one of the following subtypes: 1of1=the only free throw of the trip to the line; 1of2=the first of two free throw attempts; 2of2=the second of two free throw attempts; 1of3=the first of three free throw attempts; 2of3=the second of three free throw attempts; 3of3=the third of three free throw attempts; unk=unknown what the free throw sequence is
- made2, miss2 - a two-point field goal was made or missed, with one of the following subtypes: unk=unknown type of two-point shot; dunk=dunk; lay=layup; tip=tip-in; jump=jump shot; alley=alley-oop; drive=driving layup; hook=hook shot; stepb=step-back jump shot; pullu=pull-up jump shot; turna=turn-around jump shot; wrong=wrong basket
- made3, miss3 - a three-point field goal was made or missed, with one of the following subtypes: unk=unknown type of three-point shot; jump=jump shot; stepb=step-back jump shot; pullu=pull-up jump shot; turna=turn-around jump shot; wrong=wrong basket
- jumpb - a jumpball was called or resolved, with one of the following subtypes: start=start period; block=block tie-up; heldb=held ball; lodge=lodged ball; lost=jump ball lost; outof=out of bounds; outrb=out of bounds rebound; won=jump ball won
- X, Y - for games where it is available, this describes an X/Y position on the court where the lower-left corner of the full court is (0,0), the upper-right corner of the full court is (100,100), the exact middle of the full court (where the initial jump ball happens) is (50,50), and so on. The X/Y position is provided for fouls, turnovers, and field-goal attempts (either 2-point or 3-point).

# Modeling/Results

Once I was able to understand how the data is arranged across the various .csv files I began to create a basic model using the seeds of each of the teams and the difference between the seeds as my features.  I loaded the Teams, Seasons, Seeds and Season Results .csv's as pandas.

I used the DayZero and DayNum to create the actual date the games were played.

The tournament results from years 2015-2019 are what we will be testing our model on so I created a "test" using the 'MNCAATourneyCompactResults.csv'.  

The format for submission to kaggle has to be in a very specific format.  There can ONLY be an 'ID' column and a 'pred' column.  So the next thing I did was to create the ID column in the required format.

The ID column is a 14-character string of the format SSSS_XXXX_YYYY, where SSSS is the four digit season number, XXXX is the four-digit TeamID of the lower-ID team, and YYYY is the four-digit TeamID of the higher-ID team. Pred - this contains the predicted winning percentage for the first team identified in the ID field

For example, if you wanted to make a prediction for Duke (TeamID=1181) against Arizona (TeamID = 1112) in the 2012 tournament, with Duke given a 53% chance to win and Arizona given a 47% chance to win. In this case, Arizona has the lower numerical ID so they would be listed first, and the winning percentage would be expressed from Arizona's perspective (47%).

After this I added the seed of each of the teams and then created a seed difference column.

After all this was done I applied a Logistic Regression model on these features and got a log loss of -0.5534114526355327 which put me in top 60% of all submissions.  Not great, but not terrible for a first attempt.

After this basic submission I created a second submission using more features, more models, feature engineering and gridsearch my results were no better.  In fact they were all worse:

- Best Gradient Boosting Classifier: -0.5601615817849375
- Best Random Forest Classifier: -0.5947186182889848
- Best K-Nearest Neighbors Classifier: -0.5617566494318149
- Best Support Vector Classification: -0.5485366404702253
- Best Logistic Regression: -0.558641129132036, w/best C: 10.0

# Recommendations

---

While not completely disappointed with my Stage 1 submission I wonder if I were bring in some outside "power rankings" from the likes of Pomroy, Sagarin,, Massey Ordinals or incorporating an ELO score into my modeling might have brought me a beter result.  

Or maybe the "perfect bracket" is truly impossible and elusive....


```python

```
