#CV Coherence from Undergraduate research
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bd50b4af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import nltk\n",
    "from nltk.corpus import stopwords \n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "import string\n",
    "\n",
    "stopwords = set(stopwords.words('english'))\n",
    "punctuation = set(string.punctuation) \n",
    "lemmatize = WordNetLemmatizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "544b5bfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import codecs\n",
    "with codecs.open('P4Q1_clean.txt', \"r\",encoding='utf-8', errors='ignore') as fdata:\n",
    "    P4Q1 = fdata.readlines()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "932fa602",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['I believe that we are all from cultural backgrounds that favor out going personalities andassertive communication techniquesI believe as individualshowevermost of us are morereserved and may find it rude to take control of at he communicationWhenever one of usbecomes the most assertive communicator in the groupthe other members tend to follow theirdirectionbut input is respected\\r\\n',\n",
       " 'I do see some effects of culture in our teamwork I see some members being direct when saying theyagree or disagree with something and some members like myself being indirect with saying we disagreewith something Some members are very dominating and others just staying passive\\r\\n',\n",
       " 'Our team has not been influenced by culture yet We need tocommunicate more and show respect to cultural differences\\r\\n',\n",
       " 'I tend to be direct with what needs to be discussed I would rather get together and knock it all out as agroup That method provides inspiration by watching your team work simultaneously to finish onemilestone The last meeting with the team only two other members showed up and discussed what isneeded to get done Along with have multiple tools to communicate with each other\\r\\n',\n",
       " 'I think that culture has not necessarily played an impact in communication styles as wereall pretty proficient in the English language and have started to figure out how to cometogether properly as a group Culture does show in the fact that it seems that all of us arefrom various different countries so we may adapt different schedules in our lifestyle andhave to converge properly in order to complete tasks\\r\\n',\n",
       " 'I think culture changes how we speak with others especially with things like tone slang when wespeak and whether or not we interpret verbal ques We often have different voices when we speak todifferent people We might also use different words When t comes to our group some members voicesmay be higher or lower when they speak compared to when they speak to their family Some of usmight also use different words or phrases as they are not sure that the group will understand them Thiscould then likely impact ones ability to communicate their ideas to their team as their normalvocabulary is restricted\\r\\n',\n",
       " 'I believe culture has fostered a positive communication environmentfor my team All of us come from slightly difference backgrounds so ithas been interesting to see how each team member interacts withanother Communication for me is extremely important because I comefrom a technical theatre background where clear concisecommunication is extremely important I have been made aware notfrom this group project but from other group communication in workexperiences that my attempt to be direct and concise can come off assnoody or rude to others Because of this I always try to read theroom when I am working with a group and adjust my communicationaccordingly\\r\\n',\n",
       " 'The role of culture has a great impact on how teams work together In group 22 we havesome members who prefer to speak up on issues and others who prefer to wait until thequestion is passed to them to answer We have members who prefer one form of meetingor communication over another Our group has communicated well for the most part butthere most certainly could be issues in other groups where one persons culture is to beloud and direct and another group members culture sees loud and direct as disrespectfuland brash\\r\\n',\n",
       " 'Based on the conversations Ive had with my teammates since starting the project Ibelieve that we all come from a culture of equity So far there hasnt been any form ofhierarchy or someone who has designated themselves leader For the first submissionwe worked to divide the work equally and gave each other advice when someoneasked This has resulted in an effective work environment through the first 5 weeks ofthe project There havent been any causes of conflict so far however I believe thatwe will be able to work through it efficiently and constructively\\r\\n',\n",
       " 'I think the role of culture has played a role in our team by we share leadership for the most partand consider each other as having an equal stake in the partnership and the ideals We camefrom a similar background and that has made communication within our team relatively easy\\r\\n',\n",
       " 'I dont think that culture has had much influence on our communication styles We allappear to have similar backgrounds and communicate in similar ways as well\\r\\n',\n",
       " 'I think culture has brought a small difference in our communication roles but not a extremelysignificant one since we all seem to be from relatively similar upbringings\\r\\n',\n",
       " 'My group has not had the best communication which may be influenced by the five of us beingfrom different places or in one persons case being a bit older than everyone else Ourexperiences are very different from each others so learning how to talk to each other has beena bit of a process\\r\\n',\n",
       " 'As a team we have communicated onlywith teams and groupme I think since 2 of our members are of Asian culture Text and teams areprobably best for them as their English is not their first language It helps everyone tounderstand what they are trying to communicate and overall improves collaboration\\r\\n',\n",
       " 'For the most part our group is culturally similar or that I could tell I am probably the only onewho is more distinct I am Mexican American but I lived in a small community of mostly WhiteAmericans in Indiana for most of my entire life So we were able to communicate just fine andget along with each other No one in the group has really used any different types of slang orspeaking mannerisms that make me think they are from outside the Midwest\\r\\n',\n",
       " 'For me I rely on constant communication and precise communication I also rely on people holding trueto their word I always grew up where if you say something you would do it So when my teammatessay they will do something I trust that they will get it done\\r\\n',\n",
       " 'The cultures of our team members are not greatly dissimilar but there are definitely somedifferences specifically in regard to communication\\r\\n',\n",
       " 'I think that culture has influenced our process within our team is our work ethic Some peoplein the group come from a culture that prides itself with work ethic and getting stuff done earlyThis leads to some coming off really strong when communicating to the team because theywant to get stuff done early rather than late\\r\\n',\n",
       " 'I dont think our group is very different in the communication styles we grew up withTwo of our team members are from within an hour of purdue and two of us are from closeto the same place in the United States We were all generally raised on the same moralcode and treat one another with respect To this regard I dont think there has been a roleculture has influenced in our project\\r\\n',\n",
       " 'Culture is learned from childhood through adolescence Like in Zora Neale Hurstons essay HowIt Feels To Be Colored Me culture is only recognized when someone is removed from theirculture and consequently exposed as a minority in another culture From the outside ofperspective of a minority and through the inside perspective of the majority culture isunderstood\\r\\n',\n",
       " 'I dont know the exact culture of my team but it seems like a couple people dont want totalk either because of their home culture or their social culture and one person inparticular likes talking to everybody politely and likes working as a team I cannotconfirm if this is a culture aspect but it seems like it\\r\\n',\n",
       " 'Culture has influenced our team as each of our group members come from a differentcultural background It has influenced our conversation and how we each try to interact witheach other as we each have different views on how things should work\\r\\n',\n",
       " 'I think within our team the role of culture hasnt really influenced a lot of what we have beendoing When it comes to doing work or a job as a team no matter what culture you are frommost people want to get the job done as well and as efficiently as possible\\r\\n',\n",
       " 'Within my team communication is a big struggle that is being worked on atthe moment During the first milestone we would do work on calls many times butthere were many moments during that call when there would be silence and no onewould say anything until someone picked the conversation back up again I believethat this has a lot to due with cultural differences between all of the teammates Asstated in the video culture has a lot to do with how someone looks at the worldaround us and every person looks at the world differently Within a group there isa certain balance that has to be established in order for a group to function welltogether In this balance there are people who take control of a conversationthose who are soft spoken and those who only speak when they feel their voiceneeds to be heard In my group the main reason for our lack of communication isthese cultural differences in how people communicate and in relation how thegroup as a whole is able to communicate I believe in order for my team to becomesuccessful in communication those who are more outspoken in a conversationneed to be able to pull those who are softspoken into the conversation soeveryone can contribute to the team\\r\\n',\n",
       " 'I believe that the role of culture has positively influenced my team so far Because most of ushave similar cultural backgrounds we are all more comfortable to share our thoughts andfeelings with one another and are able to work more efficiently Though some cultures may thinkits rude to speak directly our team has shown to all be for speaking openly in order to makesure we do things correctly\\r\\n',\n",
       " 'I think that culture has influenced the communication styles within our group Some ofus are a lot more outspoken about problems we are facing or the project in generalwhereas others are more reserved This just has to do with the experiences each of ushave had previously when working with groups or how we communicate with others\\r\\n',\n",
       " 'Hasnt really influenced our process\\r\\n',\n",
       " 'Its hard to say how culture influences my teams communication because we are all frompretty much the same culture\\r\\n',\n",
       " 'I feel that we learn our own culture by finding people who share our beliefs and what wefind normal on a daily basis Culture is based around someones core values and thatshow we understand cultures of other people around us By comparing us to others inturn portrays our values To understand our values we must connect with other people tofind out for ourselves what we believe and how we should interact around each other\\r\\n',\n",
       " 'I think that culture has a significant influence in our communication style but it has always beenpositive I think our group is extremely respectful and we have not had issues communicatingthrough our milestones The influence is there and if anything it is subtle\\r\\n',\n",
       " 'Since all my group members share a similar culture ourcommunication styles are also similar This benefits our groupdynamic because there are no clashing communication styles Weall feel like we can speak directly whenever needed\\r\\n',\n",
       " 'I think the role of culture has not had a very large effect in my team most of my team is from theMidwest so as to what Dr Verghese said in his video we dont see our culture when we are inside of it\\r\\n',\n",
       " 'I am not sure how the cultures of my teammates have affected how we communicate but Idefinitely see a difference between myself and them I tend to try to communicate my problemsor how I plan to contribute to a milestone as quickly as possible I have noticed that some of myteammates will not communicate at all or communicate in fragments They also might not feelcomfortable communicating the problems that they have right away\\r\\n',\n",
       " 'I feel that the communication within our team is influenced well by culture every person in our groupgrew up either in different states or different counties Every place has its own influence of differentcultures beliefs and groups We all think differently in some way which itself can lead tomiscommunication or different expectations It works well to understand this as the group can simplylay everything out thats expected and evaluate together\\r\\n',\n",
       " 'I think the role of culture has a big influence on ourteams communication process Each one of us wasobviously raised differently but we all have around thesame college culture I think it effects our communicationbecause it depends widely on how motivated each of usare to do our assignment Talking about college culturewhere we each live and what kind of social life has majoreffects on our communication because some of us aremore motivated to get our work done then others\\r\\n',\n",
       " 'In my team I can be pretty safe to assume that all of us have been raised in anAmerican culture where a shift is being observed of more confrontation and lesshierarchy These two concepts can be observed in our team as we will interjectwhen possible in order to add more to each thought and we will treat each otherequally\\r\\n',\n",
       " 'Our team mostly communicates over GroupMe I believe that our group feel mostcomfortable to communicate in this way because we all have grown up in a culture that nonface to face communication methods are heavily relied upon\\r\\n',\n",
       " 'I think the role of culture has influenced the communication process of our teamsomewhat lightly The four group members in my group are all from the United Statesand so we all have the same groundwork cultural understanding of one another Therehas not been any type of communication issues between our group so I think the impactis light due to our similarities\\r\\n',\n",
       " 'think culture hasnt played too large of a role in my teams communication I think thatall of us talk to each other with a respective vocabulary and tone so we havent had anyissues with interfering with someones cultural values\\r\\n',\n",
       " 'As Americans we all communicate very directly with each other because we do not see it asrude We will make sure to be polite but let each other know when we disagree\\r\\n',\n",
       " 'The biggest cultural difference I see in my team as far as communication goes is theteam group chat From my experiences in Mexico school group chats were much moreinformal My current teammates Group Me is solely for the purpose of asking questionsand organizing meetings Many times from my perspective I believe we do notcommunicate as much as we should Not all questions get answered and not all areanswered in a timely manner This could be cultural or just how my team is As statedbefore in Mexico my school group chats were more informal We sometimes talkedabout other things other than school and messages were answered quicker We werecloser as a team but we definitely got more sidetracked These are two extremes that Ithink are influenced by culture I feel like I participate the most in group 9s group chatand possibly more than my other teammates are comfortable with As a team we mayneed to all adjust our communication expectations to find a middle ground to workbetter together\\r\\n',\n",
       " 'I believe culture has played a small role in influencing the communication process within my team Itshows in the way people tend to speak up Some of us talk lead like me and guide the discussion whileothers sit back and listen and only talk when needed or have an idea of their own\\r\\n',\n",
       " 'I think that our mostly shared culture has helped our team to work together well andcommunicate effectively\\r\\n',\n",
       " 'Most of the teams I have been in contained people that share a similar culture to mine so we allgot along well and enabled us to quickly get past the awkward stage I found that because of thisculture similarity we handle group problems the same and generally handled issues smoothly\\r\\n',\n",
       " 'A common way of practicing groupwork in schools is to assign each member a job andonly talking about it once everyone is done We are not the most talkative with each otherwhen not on a call as we all understand the jobs need to be done I think this just stemsfrom typical American school experiences that have influenced us to rarely ask eachother questions\\r\\n',\n",
       " 'The role of culture has influenced the team process in many ways The wayeach team member talks and communicates with the group resembles theirculture While our team has many different cultures we are able tocommunicate with each other very well and I believe that each member of thegroup understands each other and their culture and because of this there is amutual respect between us all\\r\\n',\n",
       " 'The role of culture influenced the process with my team in that we have a sharedunderstanding of what type of communication works best with each other Because ofthis consistent cultural experience I feel that communication task delegation andteamwork comes easily to us\\r\\n',\n",
       " 'I think that most if not all of my team members are from the US and within that Indianaand California From what I can tell so far we have very similar communication styles We areall able to voice our opinions and ideas for helping the group or solving problems without fear ofjudgement or that were stepping on toes To be honest I am quite happy with the group I havebeen assigned to We all work extremely well together and know that when someone isassigned the leader of the group scrum master that person is in charge of decision making forthe groupWhen going through the scrum masters they have made decisions based onconferences with the group so that a collective decision can be made We have also steppedback from trying to lead or take control when the scrum master is present showing respect forhierarchy within the group and taking note of ones place\\r\\n',\n",
       " 'I believe that culture has not really influenced the process of communication within myteam Nobody really communicates wildly differently than one another In a sense theway that it has influenced the communication of my team is that since we are similarculturally we are easy to communicate with Everybody is sort of on the same page withone another\\r\\n',\n",
       " 'Mostly everyone in our group is laid back I work well by myself This is because theculture in my household revolved around self sufficiency Our group does well to dividework and complete tasks without much collaboration Our communication is gettingbetter however it is apparent that we all prefer and work well on our own\\r\\n',\n",
       " 'I dont think the role of culture has played too much of a part within our team We do havedifferences in work styles due to us doing things differently throughout our past experiences butwe have no problems with communicating with one another to get things done\\r\\n',\n",
       " 'We do not see our own culture It is passive while we are inour version of normal\\r\\n',\n",
       " 'I think it has affected our group somewhat as most of us haveall been raised in a culture where we need to get things doneon time and efficiently We all have similar goals which is tothoroughly complete our work and with this knowledge use itto equally complete the work I think we all have the samegoal to not let the team down and combined with our workethic we use this to create the best project that we can\\r\\n',\n",
       " 'American individualism can have a huge impact on the way teams communicateinternally People are less likely to regard themselves as a collective and moredelegation and emphasis is devoted to personal responsibilityunderstanding\\r\\n',\n",
       " 'I think that the role of culture has influenced our teams communication alittle less than some other teams We do have an even split of men andwomen so I dont feel like there will be an issues with people not beingheard We all try to make sure that everyone speaks around the sameamount and that everyones opinions and ideas are taken intoconsideration\\r\\n',\n",
       " 'The main way of understanding your own culture is to immerse yourself andexperience othersthrough seeing the differences that other cultures have compared to you you canbegin to noticespecific cultural values that have become second nature\\r\\n',\n",
       " '1 I think the role of culture has influenced our team in a veryinteresting way After our incident with our team we wereable to buckle down and ger things done We were able todivide up the work equally as well as complete the worktogether as a team\\r\\n',\n",
       " 'I think the culture is a big fact that impact the process of a team because culture willaffect a persons action and mind Everyone has different mind of thinking and thoughtbased on their different culture For example in our team Asian have rigorous mindAmerican have open thoughts It will affect the team effective\\r\\n',\n",
       " 'I think culture definitely plays a role in this process within ourteam because everyone has a different communication styledespite the fact we are close in age and studies the same majorwhile being in the same college Although I dont know thespecific cultural background of my team members I could inferthat someone came from a culture were speaking up andspeaking directly is valued as strong leadership but someoneelse came from a culture that feels the opposite\\r\\n',\n",
       " '1 I dont think the role of culture has impacted out team as much as you or someone else mightthink The definition of understanding our own culture comes from the fact that we are taughtthings by other people to fit in and learn the ropes of how the people around us act and doKnowing these facts I think the process by which each of the individual teammates in my groupare influenced by culture varies very little when it comes to the process of competing work forour team\\r\\n',\n",
       " 'I dont think that the role of culture has had a significant influence on our team If anything the cultureof how we communicate over teams or online platforms speaks a little towards our culture Instead ofchoosing to meet inperson we decided especially in these times to meet over an online platform\\r\\n',\n",
       " '1 I think that within my team culture has a role but not a large one All of us are Purdue students andare quite accepting and openminded There are some differences in personality that could be culturalbut it is hard to say Some members are more quiet others are more outspoken but everybody has avoice and provides their input\\r\\n',\n",
       " 'I think most of our team communication hasnt had much impact from our culture The teamtries to communicate to get things done even though sometimes we lack full communicationWe have still completed tasks so far\\r\\n',\n",
       " 'I think the role of culture has influenced the communication styles in my team by creatingsomewhat of a barrier The background of where we all come from and things weveexperienced holds us back from being as efficient as possible in collaborating We get thejob done but I feel like we could improve by first understand one anothers culture\\r\\n',\n",
       " 'We understand our own culture like a fish in water We behave like those around us and theculture around us The second we leave that by moving or being exposed to other cultures wecan understand and reflect what makes up our own culture and the way we behave within it Youlearn that culture and it makes you like things differently to things to other people who arebiologically the same\\r\\n',\n",
       " 'the different cultures of my group members definitely could affect the way they communicate with me one of my group members could disagree with what Im saying but might not say anything because in his culture it is rude to do so Though I am not sure if this is happening or not in my group this is something that definitely can happen within group projects\\r\\n',\n",
       " 'As a team we have yet to reach a point of disagreement that will test that aspect of ourcommunication styles Other than that we communicate somewhat directly Our team makessure that we are all on the same page However we do lack real leadership at times We allseem to come from similar backgrounds which is one of the reasons we have yet to have anyserious conflict\\r\\n',\n",
       " 'I think the fact that my team has an open and welcomingculture is part of the reason our communication style hasdeveloped to be so free At the moment I feel like anyonethinks they can contribute and wont be ridiculed for thoughtsthey have or found to be a detriment if they make asuggestion that is rude Furthermore everyone knows theycan speak directly to each other and dont have to worryabout any hierarchical struggles which means that ourmeetings and teamwork is much more fluid and constantlymoving\\r\\n',\n",
       " 'I believe the role of culture has influenced my team in terms of leadership Thereare many different aspects of culture within a single culture For example culturein the US can be broken down into several categories You can separate the northfrom the south and the east coast from the west coast In the team you can tellwho is a strong leader and who does not want to lead and just wants to do workThis is the biggest aspect that I have seen culture manipulate within my team\\r\\n',\n",
       " 'Culture has been a large focus of how we communicate in a group We wantedto create a healthy and communicative group culture for our projects before we startedworking We have decided to be upfront with each other as much as possible so wedont have any unheard complaints or criticisms By everyone understanding andrespecting the culture of our group we are able to work more effectively and produceour best product\\r\\n',\n",
       " 'I think that within my team the role of culture has influenced me and my otherteam members to communicate with each other with respect and being open with what ishappening on our ends in relation to the work for the project\\r\\n',\n",
       " 'Since all of my team members are also American I think we all have a similar mindsetwhen it comes to our culture and how we communicate with one another Us being allAmerican has made the team process much simpler since I dont need to worry aboutproperly getting my point across or addressing cultural differences that might have beenapparent with someone who didnt grow up in America\\r\\n',\n",
       " 'In the video Dr Verghese explains that culture is a lens that influences how each individualviews the world and it has indeed influenced the process of communication heavily An exampleis how in my culture I see the perfect time for lunch as 12pm and all my other colleaguesbelieve 12 to be a more appropriate time and thus 122 can never be an option Dinner time isalso different in that one of our colleagues eats dinner from 56 instead of 67 from us Thus 57does not allow for any of us to work\\r\\n',\n",
       " 'Surprisingly culture has played a quite large role in influencing the way we communicate Oneof our group members is over double my age and because of that he did not grow up in thesame internet age as me and the other group member did Because of this often I prefer to useDiscord and such to talk with the one groupmate while when all working together we must useSMS messaging as it suits our older groupmate much better This leads to some interestingmoments where since I almost never use SMS messaging I can be late to seeing a message sinceI am not used to it at all\\r\\n',\n",
       " 'I believe that the role of culture presents a sort of medium of communication that allowsfor the disclosure of diverse ideas between teammates However I dont believe thatculture has affected our team in any significant way\\r\\n',\n",
       " 'Overall our team has good communication as to working through milestone 1 The roleof culture has an influence on teamwork communication because how we grew up andfactors such as family background ethnicity religion identity etc affects how wecommunicate with each other and our way of expressing our message or get our pointacross Communication patterns of individuals differ with culture which is why it plays abig role in teamwork Our mindsets and how we communicate differ because we havedifferent cultures For example while someone may think that interrupting is rude it canbe a normal thing for another group member\\r\\n',\n",
       " 'From my perspective the role of culture has negative effect in communication Forexample English is not my first language so sometimes I cant express my ideaclearly What is more the method we solve problem is different too based on cultures\\r\\n',\n",
       " 'I think my team is following a consistent style of communication We areusually direct and into the point I do not think there is a big influence ofculture in my team because my teammates are all from the US I am not fromthe US but I have been here long enough to get the culture and practice itTherefore we are all following the same culture\\r\\n',\n",
       " 'The team I am apart everyone seems pretty comfortable with American culture because the guyin the video said you dont see your culture in till you are out of it so since I havent thoughtabout this up in till now my groups culture is similar enough that I havent noticed anything\\r\\n',\n",
       " 'Since our team has members from different parts of the world there iscertainly a difference between communication styles and how each person sharestheir thoughts about the tasks I believe that the role of culture did affect ourteamwork but not necessarily in a negative manner Since me and one othermember of our group are Arabs and the other two are American you can sensehow there is a different way of communicating Some of us dont like to speakdirectly especially when there is a point of conflict While others prefer it to bethat way We got used to it since we are a good team who respect each other butit does make it seem that there are differences Im someone who was raised tonot be strictly direct as it is considered a bit rude so I dont do that But since Igrew up with others from different cultures I understand it and dont consider itto be rude when it happens to me\\r\\n',\n",
       " 'I have noticed that my partner Tulsi is definitely a lot morenot hyper but just full of energy compared to the rest of usI seem to have a bit of a seriousto the point approach with asmall blend of small talk Nate and Kyle are definitely on thequiet side I would honestly say Nate is even a little stoickConsidering all of this I think I have managed to remainculturally understanding Every culture encourages differentbehaviors and ideasI have always tended to be the welcoming and understandingone in the group because I feel that diversity to an extent brings an asset to a team Every culture has differentcommunication styles which means they also have differentways of thinking which can aid in problem solving\\r\\n',\n",
       " 'I think that the role of culture has chosen who takes charge during our meeting times We have 3males and 1 female in our group the people who are speaking most in the group are the males Weneed to be aware that Alice has great ideas and need to give her the opportunity to share those ideasHistorically culture has said that males are the leaders and that is just not the case\\r\\n',\n",
       " 'The role of culture has influenced our communication styles within our team becauseeach one of us has learned different growing up For example I may look at a problemdifferently then one of my teammates because we take a different approach However itis important to take these different approaches because we may catch things otherswouldnt see and communicate that within our team\\r\\n',\n",
       " 'I believe the role of culture positively influences teamwork It creates different thinking amongthe group that can be used for problem solving A team full of people in different cultures maybe better problem solvers than people that all belong to the same culture Different mindsetscan be important in problem solving\\r\\n',\n",
       " 'I believe that culture influenced my team to respect each other and give everyone thechance to speak up We are 4 team members 2 are from the US and 2 from Saudi Arabia so itreally balances well\\r\\n',\n",
       " 'One of the first things my team did when we got together for our first meeting wasintroduce ourselves including where we were all from One member of our team is from Indiaone member is from South Korea and the remaining two members are from the United Statesof America Two members of my groups primary language is English while it is a secondarylanguage for the other two members which can cause confusion going both ways\\r\\n',\n",
       " 'The role of culture has influenced the communication process in our team in several ways Thefirst way is since we are all from the same ethnic background we are comfortable talking toeach other and do not have to worry about talking to others in a position of authority Secondsince we all cam from the midwestern United States we all share the mannerisms associatedwith that region and thus have an easier time communicating with one another nonverbally\\r\\n',\n",
       " 'I am unsure how my teammates cultures influenced their process with our team but for me Igrew up treating everyone as an equal I listen to anyone that wants to speak their mind withinthe group I think its important to hear everyones point of view and to not push my own viewsover anyone else\\r\\n',\n",
       " 'My team tends to talk about a lot of topics unrelated to the workwhile we are working together These conversational topics arerelated to the other teammates cultures because my group hassome diversity in culture such as American Korean and Indian Whilethis conversation may seem distracting at first it helps theteammates feel more friendly and open to giving ideas I believe thispractice is based upon American culture which is formed by peoplewith various backgrounds and it is natural to be embracing and beingcurious about the differences\\r\\n',\n",
       " 'Within my team I think that the role of culture has influenced our process of communication inmany different ways Since we are all pretty new to collective teamwork in a college setting besidestech 120 we havent really had these opportunities to look at different situations through anotherpersons perspective Even though we might share many qualities in our day to day lives we still bring adiverse group of opinions and work ethics to the table In my team we have a student in the militaryfraternity different classes same classes older younger etc Each of these perspectives gives us anadvantage an example would be that some people have greater social skills than others and are able tostructure our meeting times so that we are able to stay on track include everyone in the groups ideasand be able to stay on the right path to complete a goal Without these different ways to communicateideas and opinions I think that we would be at a great loss Being said since all of us come fromdifferent places there are times when our communication could be improved Ive noticed that thereare more type A personalities than B and this can sometimes create an awkward communicationbecause some folks might interrupt or talk over others with or without meaning to On the other handthose who are more type B might have great ideas but dont always like to share what they are thinkingabout and this can put a dent in our communication progress In conclusion the role of culture hasgreatly given us an advantage as a team because of the different backgrounds strengths and even ourweaknesses It is important to consider these advantages and even disadvantages in every meetings sothat we can better utilize these skills to create a superb project\\r\\n',\n",
       " 'I think the backgrounds each of our team members comes from changes the way each of uscommunicate with each other It is interesting to note since I think all four of us each have adistinct way of engaging in team communication Some of us are quieter and only speak whensomething needs to be said Others of us are talking more often sometimes just to get the teammembers to know each other better I think this represents a difference in cultural backgroundwhere some people may have learned that it is more important to get the work done firstbefore engaging in social conversation Another example is when we discuss responsibilitiescertain members are willing to tell the team that they could experiment with something theyhave not done before while others prefer to understand the task before deciding they want totake it I think this also feeds into the risk calculus of different cultures Some people may learnthat trying new ideas is the way to move forward while other people may be taught that theyshould focus on what they know can be done\\r\\n',\n",
       " 'I think that culture plays a huge role in how people view time whichaffects how people communicate Some cultures are much less focusedon time and more relaxed about deadlines If an assignment is finished bythe due date then when you completed it doesnt matter On the otherhand some cultures are very timefocused and accomplishing taskssooner in order to give yourself more time is highly valued It seems likethis has played a role in my group specifically the clashing of these twovalues has led to conflict\\r\\n',\n",
       " 'My personality communication style and interaction with others has been greatlyinfluenced by my exposure with and time spent in the military culture I am stillworking on trying to get along with people that have not been exposed to that kind ofwork environment\\r\\n',\n",
       " 'I believe that most of my teammates are from the United States where the Western value ofbeing forward focusing on getting the work is relevant We express our ideas straightforwardlyand I feel that that is effective communication\\r\\n',\n",
       " 'Some members have a very takecharge attitude which isgreat because we get things done but sometimes certainmeaningful suggestions or voices go unheard rarely though\\r\\n',\n",
       " 'Its difficult to say if there has been a significant effect of culture on my teams interactionsSince I dont see them face to face I dont have a strong grasp of what their general culturesmight be As it stands we havent had any problems with communication due to differentcommunication cultures\\r\\n',\n",
       " 'Depending on countries the way to express own opinion and the wayto pursue it seems to be very different\\r\\n',\n",
       " 'I believe most if not all of the members including myself of our team grew up withinAmerican culture so that culture has influenced communication within our team themost As peers we give each other respect and allow everyone to speak whatever is ontheir mind\\r\\n',\n",
       " 'I do think some us tend to speak a it less because we are taught to be quitter inpublic But that has become less and less as we continue to work together andthe work bond has increased One thing that all cultures are taught and we followis waiting until one person finishes talking before the next person starts talkingThis way none of us feel talked over\\r\\n',\n",
       " 'I think that as mostly Americans we use distinctly direct forms of communication Wheneversomeone is behind on work or their work is due soon we will mention them and try and gatherinformation about how theyre coming along as to ensure that itll be done by the due dateSome people are afraid to speak up though because they might interpret it as rude I dont thinkthis is due to culture but rather upbringing\\r\\n',\n",
       " 'For our team we get four cultures since most of us are from differentcountry More than that even the two members from the same countrylive in different area I was one of the two members so that I knowthere are still some invisible obstacles that distinguish us We two areboth from China which occupies a relatively large territory with manyseparated provinces and regions Weve been learning differentsubcultures since we were born Fortunately we have always had avery pleasant meeting every time I cant say Im familiar with everyculture that my team contains but they seem to be very coordinatedso far Some members tend to be more active and some members tendto be more serious during the teamwork The good point is that all ofour members are trying to protect the harmony between our differentcultures We are trying to understand and respect the cultures ofothers and we are also trying to share good aspects of our cultureswith others For instance members behave more actively would try toinitiate the conversation with the passive members and make theminvolved in the teamwork Some members like to talk about somefunny stuff or things that are irrelevant to the current work If thathappens others will not seriously interrupt them instead they justwait patiently or even join the conversation I think teamwork give uschance to know about other cultures to respect other cultures tounderstand other cultures and to enjoy other cultures Reversely therole of culture has made our working process more harmonioussmoother and more comfortable It actually improved team efficiencyin some degree\\r\\n',\n",
       " 'I think that the role of culture has influenced the communication styles within our own teamworkheavily Although each of us may have had a similar upbringing and thinking about situations there arestill some cultural differences that influence communication styles and help when solving problems werun into In our group we respect when other people are speaking and listen to what they have to say Ifwe disagree with something then it is brought up and we provide an alternative viewpoint We seemedto have formed a process that relies for organization from the scrum master and then feedback from theother members of the group We also make sure that everybody agrees on a topic before we proceedwith the actual work I think that all these decisions come from a cultural influence\\r\\n',\n",
       " 'I think that culture has definitely influenced the communication process in our groupEach of our group members come from different places and different households Ourcommunication and the way we interact has been tainted by the ways we were raisedRight now the biggest culture impact on our group has been the time difference betweengroup members Three of us in our group are in EST and the other two are in differenttime zones One of our group members is from Dubai so the time difference is prettylarge We have been communicating and working together to make sure everyone is onthe same page and can meet when we get together as a group The time difference isan easy barrier to overcome as long as everyone works together and communicates\\r\\n',\n",
       " 'I think we all have a similar culture so Im not sure if our work has been influenced too much\\r\\n',\n",
       " 'People come into teams with different backgrounds and culture so it will have animpact on communication styles I know that all of my team members are fromdifferent cultures but I think because we are all students and we all have acommon goal of doing well in the class we know how to adapt to each other andlearn each others cultures in order to do work efficiently\\r\\n',\n",
       " 'The role of culture is to learn from others In my own country thecommunication style is different from the culture in the States People arenot encouraged to speak directly to other however it is different from thenew culture that I learned The common way to present our own culture isnot directly To help the process within the communication we have therole leader to start the communication and work along the entireteamwork\\r\\n',\n",
       " 'I think the role of culture has influenced the way our team communicates with eachother Each one of us has a different background Since we were grouped by time zonewe all are relatively close to each other One of us is in Korea while the remainingthree are in China Although we are all Asian like the video mentions there are manysubcultures In our case I am Chinese but I went to elementary school in Connecticutand continued the rest of school in an international school in Shanghai Similarlyanother group member has done a bulk of their education in the United States Theother two group members do not have as much western influence In terms ofcommunication style some of us like to talk more directly and have more confidencein making decisions while others do not speak as much and do not jump to makedecisions for the team In this sense I think people who go through the westerneducation system are usually more direct and blunt\\r\\n',\n",
       " 'I think that culture has influenced the communication styles in my team inmy ways Since some cultures have a more structured way ofcommunicating some members of the team treat each meeting in a veryformal structured way People from other cultures treat the meeting as casualand open Each member speaks freely and openly about problems that ariseThese communication style differences arise from team members ofdifferent cultures Another example of how culture has influenced mygroups communication style is how conflicts are handled In European andAsian countries disagreements tend to be more personally threatening thanin western culture United states This enhances the communication in mygroup because people handle disagreements differently These are just twoof the many ways that culture in my group influences communication styles\\r\\n',\n",
       " 'I was the only Korean in the group and the rest were Chinese but I didnt feel so differentbecause half of us were culturally naturalized to the States and we had a common ground of knowinghow to deal with different cultures\\r\\n',\n",
       " 'Our group is open to providing suggestions and improvements to our work which I believe canhelp us communicate effectively and help the team create a better overall project that we are allsatisfied with\\r\\n',\n",
       " 'As many teammates learned the American culture we ought to be individualistic rather thanrelying on others Obviously when there is something I do not understand I would ask thequestion and my team would reply However if I were to push my work on to the teammatesthey would not allow it The teams communication styles would be differ depending on culturebut from my culture where my life is hierarchical it would be best to speak indirectly ratherthan speaking straightforward during the team meet especially with individual older than me\\r\\n',\n",
       " 'While I was doing Milestone 1 with my team there were ChineseNew Year events going on at the same time I could hear thefireworks through my friends microphone Other than this Iwouldnt say that there were any noticeable cultural differencesthat affected the teamwork or the communication in general Sincewe are students of a global university in an increasingly globalizedworld everything is becoming standardized even people\\r\\n',\n",
       " 'I personally feel as though the communication styles within my team are verysimilar As far as I know we all come from western countries who focus onindividualism While we do convene to discuss work and compile completedassignments most assignments are done individually with minimumcommunicationThe positive side of a team full of likeminded individualistsmeans that we can focus on our portion of the work and give it our full attentionwhich in turn produces a higher quality body of work\\r\\n',\n",
       " 'Without making too many assumptions about the cultural upbringing of my team members it seemsas though culture isnt creating too much of a barrier for our group and we seem to get along well withhow we are communicating with each other\\r\\n',\n",
       " 'Because people in our team come from different countries so the communication stylesare various Some of them tends to express their opinions immediately and someoneprefers waiting until others ask\\r\\n',\n",
       " 'I think how the role of culture has influenced my communication style within my team is that inmy culture it is considered rude to speak directly when not permitted to do so For example in aclassroom setting if students are not permitted by teachers to speak we are being rude to theteachers and disruptive to the class As my culture is a highpower distance culture and everyonehas their rightful place in society subordinates are expected to be told what to do and we are alsoexpected to display respect for those of higher status This has shaped and influenced mycommunication style of not speaking up when there are problems and only doing what I am toldto do However after studying in the US and as I am out of my culture I have changed mycommunication styles significantly and have participated more in team projects by speaking upwhen there are problems and freely sharing my ideas and opinions with the whole team\\r\\n',\n",
       " 'I think our culture matters a lot to how we interact as a team Our shared culture is that ofstudents within Purdue University As students we view ourselves as equals in terms of teamwork as adefault We believe as a culture that work should be equally divided and assume that others are equallyskilled as we are until proven otherwise Compared to work within a company work among studentsis extremely democratic with leaders and assignments being temporary and either selected by the teamor selfselected\\r\\n',\n",
       " 'Culture is like our eyes Even if we have the same conditions exceptour eyes our attitude towards same things is different because what wesee is different I have been exposed to two completely differentcultures Chinese culture and American culture In the past four years Ihave lived in the United States longer than in China which gave mesome understanding of American culture\\r\\n']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "P4Q1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d74d12c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas\n",
    "df = pandas.DataFrame(P4Q1)\n",
    "def cleaning(article):\n",
    "    one = \" \".join([i for i in article.lower().split() if i not in stopwords])\n",
    "    two = \"\".join(i for i in one if i not in punctuation)\n",
    "    three = \" \".join(lemmatize.lemmatize(i) for i in two.split())\n",
    "    return three"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e5e3102d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "118"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns = ['P4Q1']\n",
    "text = df.applymap(cleaning)['P4Q1']\n",
    "text_list = [i.split() for i in text]\n",
    "len(text_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fbf3ff78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['believe',\n",
       " 'cultural',\n",
       " 'background',\n",
       " 'favor',\n",
       " 'going',\n",
       " 'personality',\n",
       " 'andassertive',\n",
       " 'communication',\n",
       " 'techniquesi',\n",
       " 'believe',\n",
       " 'individualshowevermost',\n",
       " 'u',\n",
       " 'morereserved',\n",
       " 'may',\n",
       " 'find',\n",
       " 'rude',\n",
       " 'take',\n",
       " 'control',\n",
       " 'communicationwhenever',\n",
       " 'one',\n",
       " 'usbecomes',\n",
       " 'assertive',\n",
       " 'communicator',\n",
       " 'groupthe',\n",
       " 'member',\n",
       " 'tend',\n",
       " 'follow',\n",
       " 'theirdirectionbut',\n",
       " 'input',\n",
       " 'respected']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "text_list[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d46264a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dictionary(1471 unique tokens: ['andassertive', 'assertive', 'background', 'believe', 'communication']...)\n"
     ]
    }
   ],
   "source": [
    "# Importing Gensim\n",
    "import gensim\n",
    "from gensim import corpora\n",
    "\n",
    "# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)\n",
    "dictionary = corpora.Dictionary(text_list)\n",
    "dictionary.save('P4Q1.dict')\n",
    "print(dictionary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "53888329",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "118\n",
      "[(17, 8), (19, 1), (25, 1), (27, 2), (29, 7), (36, 1), (37, 2), (38, 1), (43, 3), (48, 2), (50, 3), (59, 1), (64, 1), (74, 3), (76, 1), (77, 1), (85, 1), (104, 1), (108, 1), (129, 1), (133, 1), (143, 1), (159, 1), (168, 1), (171, 1), (203, 2), (211, 1), (223, 2), (240, 1), (241, 1), (252, 1), (263, 1), (267, 1), (268, 1), (279, 1), (280, 3), (291, 1), (310, 1), (331, 1), (365, 1), (373, 1), (405, 1), (429, 1), (437, 1), (488, 1), (504, 1), (547, 1), (572, 1), (726, 2), (793, 1), (809, 1), (821, 1), (831, 1), (836, 1), (950, 2), (960, 1), (1028, 1), (1067, 1), (1119, 1), (1122, 1), (1241, 1), (1242, 1), (1243, 1), (1244, 1), (1245, 1), (1246, 1), (1247, 1), (1248, 1), (1249, 1), (1250, 1), (1251, 1), (1252, 1), (1253, 1), (1254, 1), (1255, 1), (1256, 1), (1257, 1), (1258, 1), (1259, 1), (1260, 1), (1261, 1), (1262, 1), (1263, 1), (1264, 1), (1265, 1), (1266, 1), (1267, 1), (1268, 1), (1269, 1), (1270, 1), (1271, 1), (1272, 1), (1273, 1), (1274, 1), (1275, 1), (1276, 1), (1277, 1), (1278, 1), (1279, 1), (1280, 1), (1281, 1), (1282, 1), (1283, 1), (1284, 1), (1285, 1), (1286, 1), (1287, 1), (1288, 1), (1289, 1), (1290, 1)]\n"
     ]
    }
   ],
   "source": [
    "# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.\n",
    "doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]\n",
    "corpora.MmCorpus.serialize('corpus_P4Q1.mm', doc_term_matrix)\n",
    "\n",
    "print(len(doc_term_matrix))\n",
    "print(doc_term_matrix[100])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7cb236cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import logging\n",
    "import pyLDAvis\n",
    "import json\n",
    "\n",
    "\n",
    "from gensim.models.coherencemodel import CoherenceModel\n",
    "from gensim.models.ldamodel import LdaModel\n",
    "from gensim.corpora.dictionary import Dictionary\n",
    "from numpy import array\n",
    "\n",
    "# Creating the object for LDA model using gensim library\n",
    "Lda = gensim.models.ldamodel.LdaModel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "105d7e57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.32718313231184065\n"
     ]
    }
   ],
   "source": [
    "lda5 = Lda(doc_term_matrix, num_topics=5, id2word = dictionary)\n",
    "cm5 = CoherenceModel(model=lda5, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm5.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f8f02263",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.32382001085333834\n"
     ]
    }
   ],
   "source": [
    "# Running and Trainign LDA models on the document term matrix.\n",
    "lda10 = Lda(doc_term_matrix, num_topics=10, id2word = dictionary)\n",
    "cm10 = CoherenceModel(model=lda10, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm10.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "00cf4de8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3510608998882279\n"
     ]
    }
   ],
   "source": [
    "lda6 = Lda(doc_term_matrix, num_topics=6, id2word = dictionary)\n",
    "cm6 = CoherenceModel(model=lda6, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm6.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2d123032",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.32869851570236036\n"
     ]
    }
   ],
   "source": [
    "lda7 = Lda(doc_term_matrix, num_topics=7, id2word = dictionary)\n",
    "cm7 = CoherenceModel(model=lda7, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm7.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "db1a8c7b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3157256819249674\n"
     ]
    }
   ],
   "source": [
    "lda12 = Lda(doc_term_matrix, num_topics=12, id2word = dictionary)\n",
    "cm12 = CoherenceModel(model=lda12, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm12.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3dc8098a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.320517628798915\n"
     ]
    }
   ],
   "source": [
    "lda8 = Lda(doc_term_matrix, num_topics=8, id2word = dictionary)\n",
    "cm8 = CoherenceModel(model=lda8, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm8.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7198e25a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.327957207976931\n"
     ]
    }
   ],
   "source": [
    "lda9 = Lda(doc_term_matrix, num_topics=9, id2word = dictionary)\n",
    "cm9 = CoherenceModel(model=lda9, texts=text_list, dictionary=dictionary, coherence='c_v')\n",
    "print(cm9.get_coherence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ca46d50",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
