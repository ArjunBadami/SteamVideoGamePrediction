#UCSD FALL 2023
#CSE 258
#Assignment 1
#Name: Arjun H. Badami
#PID:A13230476

import gzip
import stat
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf

import gzip
from collections import defaultdict

def readJSON(path):
    f = gzip.open(path)
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u, g, d

def readGz(path):
    for l in gzip.open(path, 'rt', encoding='utf8'):
        yield eval(l)

data = []
for l in readJSON("train.json.gz"):
    data.append(l)

dataTrain = data[:165000]
dataValid = data[165000:]


### Time-played preprocessing - TASK 1
allHours = []
userHours = defaultdict(list)
gameHours = defaultdict(list)
userHoursTrain = defaultdict(list)
userHoursValid = defaultdict(list)
gameHoursTrain = defaultdict(list)
gameHoursValid = defaultdict(list)
pairHoursTrain = defaultdict(list)
pairHoursValid = defaultdict(list)
pairHours = defaultdict(list)
userIDs = {}
gameIDs = {}
i = 0
for user,game,d in data:
  h = d['hours_transformed']
  allHours.append(h)
  userHours[user].append(h)
  gameHours[game].append(h)
  pairHours[(user, game)].append(h)
  if not user in userIDs: userIDs[user] = len(userIDs)
  if not game in gameIDs: gameIDs[game] = len(gameIDs)
  if(i < 165000):
      userHoursTrain[user].append(h)
      gameHoursTrain[game].append(h)
      pairHoursTrain[(user, game)].append(h)
  else:
      userHoursValid[user].append(h)
      gameHoursValid[game].append(h)
      pairHoursValid[(user, game)].append(h)
  i = i + 1

hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

globalAverageTrain = sum(hoursTrain) / len(hoursTrain)
globalAverage = sum(allHours) / len(allHours)
userAverageTrain = {}
for u in userHoursTrain:
  userAverageTrain[u] = sum(userHoursTrain[u]) / len(userHoursTrain[u])

### Would-play preprocessing - TASK 2
gameCountTrain = defaultdict(int)
gameCountValid = defaultdict(int)
totalPlayedTrain = 0
totalPlayedValid = 0
allgamesplayedbyuser = defaultdict(set)
allusersplayedgame = defaultdict(set)
gamesplayedbyuserTrain = defaultdict(set)
gamesplayedbyuserValid = defaultdict(set)
usersplayedgameTrain = defaultdict(set)
usersplayedgameValid = defaultdict(set)
allgames = set()
i = 0
for user,game,_ in data:
    allgames.add(game)
    allgamesplayedbyuser[user].add(game)
    allusersplayedgame[game].add(user)
    if(i < 165000):
        gamesplayedbyuserTrain[user].add(game)
        usersplayedgameTrain[game].add(user)
        gameCountTrain[game] += 1
        totalPlayedTrain += 1
    else:
        gamesplayedbyuserValid[user].add(game)
        usersplayedgameValid[game].add(user)
        gameCountValid[game] += 1
        totalPlayedValid += 1

    i = i + 1


mostPopularTrain = [(gameCountTrain[x], x) for x in gameCountTrain]
mostPopularTrain.sort()
mostPopularTrain.reverse()

return1 = set()
count = 0
for ic, i in mostPopularTrain:
  count += ic
  return1.add(i)
  if count > totalPlayedTrain/1.5: break

def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


##################################################
# Play prediction                                #
##################################################
newValid = []

for d in dataValid:
    u = d[0]
    g1 = d[1]
    g2 = random.choice(list(allgames - gamesplayedbyuserValid[u]))

    newValid.append((u, g1, 1))
    newValid.append((u, g2, 0))


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(gameIDs)], stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs), K], stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(gameIDs), K], stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +
                            tf.nn.l2_loss(self.gammaU) +
                            tf.nn.l2_loss(self.gammaI))

    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))

optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(5, 0.00001)

def trainingStepBPR(model):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            d = random.choice(data)
            u = d[0]
            i = d[1]
            j = random.choice(list(allgames - allgamesplayedbyuser[u]))
            sampleU.append(userIDs[u])
            sampleI.append(gameIDs[i])
            sampleJ.append(gameIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


for i in range(100):
    obj = trainingStepBPR(modelBPR)
    #if(float(obj) < 0.44): break
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

numnotseen = 0
testUserGameScores = defaultdict(list)
testUserGameScoresNew = defaultdict(list)
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u, g = l.strip().split(',')

    #uA = usersplayedgameTrain[g]
    uA = allusersplayedgame[g]
    max = 0
    pred3 = 0
    pred4 = 0
    for g1 in allgamesplayedbyuser[u]:
        uB = allusersplayedgame[g1]
        sim = Jaccard(uA, uB)
        if sim > max:
            max = sim

    score = max + (0.08 * (g in return1))
    testUserGameScores[u].append((score, g))

    score1=0
    if(u in userIDs and g in gameIDs):
        score1 = modelBPR.predict(userIDs[u], gameIDs[g])
    else:
        score1 = 0
        numnotseen += 1

    testUserGameScoresNew[u].append((score1, g))


testUserGameWillPlay = defaultdict(list)
testUserGameWillNotPlay = defaultdict(list)

for u in testUserGameScoresNew:
    l = testUserGameScoresNew[u]
    l.sort()
    np = l[:int(len(l)/2)]
    p = l[int(len(l)/2):]

    np1 = [r[1] for r in np]
    p1 = [r[1] for r in p]
    testUserGameWillPlay[u] = p1
    testUserGameWillNotPlay[u] = np1

#FINAL PREDICTIONS
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, g = l.strip().split(',')

    pred = 0
    if g in testUserGameWillPlay[u]:
        pred = 1

    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()



##################################################
# Hours played prediction                        #
##################################################
stddev = numpy.std(allHours)
betaU = {}
betaI = {}
for u in userHours:
    betaU[u] = 0

for g in gameHours:
    betaI[g] = 0

alpha = globalAverage

def calc_new_alpha(lamb):
    l = []
    for pair in pairHours:
        hours = pairHours[pair]
        user = pair[0]
        game = pair[1]
        for h in hours:
            s = h - (betaU[user] + betaI[game])
            l.append(s)

    new_alpha = sum(l) / len(allHours)
    return new_alpha

def calc_new_betaU(lamb):
    for user in allgamesplayedbyuser:
        games = allgamesplayedbyuser[user]
        l = []
        for game in games:
            hours = pairHours[(user,game)]
            for hour in hours:
                s = hour - (alpha + betaI[game])
                l.append(s)
        betaU[user] = (sum(l) / (lamb + len(games)))
    return betaU

def calc_new_betaI(lamb):
    for game in allusersplayedgame:
        users = allusersplayedgame[game]
        l = []
        for user in users:
            hours = pairHours[(user,game)]
            for hour in hours:
                s = hour - (alpha + betaU[user])
                l.append(s)
        betaI[game] = sum(l) / (lamb + len(users))
    return betaI


for i in range(1000):
    print("2: " + str(i))
    alpha = calc_new_alpha(5)
    betaU = calc_new_betaU(8.1)
    betaI = calc_new_betaI(2.5)


class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb, stddev):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)], stddev=0.5))
        self.betaI = tf.Variable(tf.random.normal([len(gameIDs)], stddev=0.5))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs), K], stddev=0.5))
        self.gammaI = tf.Variable(tf.random.normal([len(gameIDs), K], stddev=0.5))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        #p = self.alpha + self.betaU[u] + self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        p = tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU ** 2) +
                            tf.reduce_sum(self.betaI ** 2) +
                            tf.reduce_sum(self.gammaU ** 2) +
                            tf.reduce_sum(self.gammaI ** 2))

    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = (self.alpha + beta_u + beta_i +
                tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1))
        return pred

    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)


optimizer = tf.keras.optimizers.Adam(0.1)
modelLFM = LatentFactorModel(globalAverage, 1, 5, stddev)

def trainingStep(model):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleH = [], [], []
        for _ in range(Nsamples):
            d = random.choice(data)
            u = d[0]
            i = d[1]
            h = d[2]['hours_transformed']
            sampleU.append(userIDs[u])
            sampleI.append(gameIDs[i])
            sampleH.append(h)

        loss = model(sampleU,sampleI,sampleH)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


for i in range(100):
    obj = trainingStep(modelLFM)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))


def finalpred(u, g, model):
    gammaprod = float(model.predict(userIDs[u], gameIDs[g]))
    return alpha + betaU[u] + betaI[g] + (100*gammaprod)


#FINAL PREDICTIONS
predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, g = l.strip().split(',')

    pred = finalpred(u, g, modelLFM)

    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()