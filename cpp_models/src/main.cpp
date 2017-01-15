#include <stdio.h>
#include <math.h>
#include "ControlAlgorithm.h" // SarsaControl, QControl, Q
#include "PredictorAlgorithm.h" // Sarsa
#include "Hashing.h"  // MurmurHashing
#include "Projector.h"  // TileCoderHashing
#include "StateToStateAction.h" // StateActionTilings
#include "Mathema.h" // Random
#include "ParamMountainCar.h"
#include "gridworld.h"
#include "RL.h"


int mountain_car(int argc, char* argv[])
{
    using namespace RLLib;

    double throttleFactor = 1.0;
    int trainingEpisodes = 1000;
    int logEpisodes = 10;
    int seed = 0;
    if (argc > 2)
        throttleFactor = atof(argv[2]);
    if (argc > 3)
        trainingEpisodes = atoi(argv[3]);
    if (argc > 4)
        logEpisodes = atoi(argv[4]);
    if (argc > 5)
        seed = atoi(argv[5]);

    Random<double>* random = new Random<double>;
    random->reseed(seed);
    RLProblem<double>* problem = new ParamMountainCar<double>(random, throttleFactor);

    Hashing<double>* hashing = new MurmurHashing<double>(/*random*/random,
                                                         /*memorySize*/10000);
    Projector<double>* projector =
        new TileCoderHashing<double>(/*hashing*/hashing,
                                     /*nbInputs*/problem->dimension(),
                                     /*gridResolution*/10,
                                     /*nbTilings*/10,
                                     /*includeActiveFeature*/true);

    StateToStateAction<double>* toStateAction =
        new StateActionTilings<double>(/*projector*/projector,
                                       /*actions*/problem->getDiscreteActions());
    Trace<double>* e = new RTrace<double>(projector->dimension());

    double alpha = 0.15 / projector->vectorNorm();
    double gamma = 0.99;
    double lambda = 0.3;
    Sarsa<double>* sarsa = new Sarsa<double>(alpha, gamma, lambda, e);
    double epsilon = 0.02;
    Policy<double>* acting =
        new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
    OnPolicyControlLearner<double>* control =
        new SarsaControl<double>(acting, toStateAction, sarsa);

    int maxSteps = 10000;
    int nbRuns = 1;
    RLAgent<double>* agent = new LearnerAgent<double>(control);
    RLRunner<double>* sim =
        new RLRunner<double>(agent, problem, maxSteps, trainingEpisodes, nbRuns);
    sim->setVerbose(false);
    sim->runEpisodes();

    delete acting;
    delete control;
    delete agent;
    delete sim;

    // simulation with learned agent
    epsilon = 0.0;
    sarsa->setLearn(false);

    acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), sarsa, epsilon);
    control = new SarsaControl<double>(acting, toStateAction, sarsa);
    agent = new LearnerAgent<double>(control);
    sim = new RLRunner<double>(agent, problem, maxSteps, logEpisodes, nbRuns);
    sim->setVerbose(false);
    sim->setEnableLog(true);
    sim->runEpisodes();

    delete random;
    delete problem;
    delete hashing;
    delete projector;
    delete toStateAction;
    delete e;
    delete sarsa;
    delete acting;
    delete control;
    delete agent;
    delete sim;
    return 0;
}

/**
 * argv[2] == "grid_world"
 * argv[3]   : int size
 * argv[4]   : int training_episodes
 * argv[5]   : int log_episodes
 * argv[6]   : int seed
 * argv[7]   : int world_seed
 * argv[8]   : int n_features
 * argv[9+i] : float feature_value_i
 */
int grid_world(int argc, char* argv[])
{
    using namespace RLLib;

    if (argc < 9)
    {
        std::cout << "argv[2]   : int size" << std::endl;
        std::cout << "argv[3]   : float prob_rnd_move" << std::endl;
        std::cout << "argv[4]   : int training_episodes" << std::endl;
        std::cout << "argv[5]   : int log_episodes" << std::endl;
        std::cout << "argv[6]   : int output_type" << std::endl;
        std::cout << "argv[7]   : int seed" << std::endl;
        std::cout << "argv[8]   : int world_seed" << std::endl;
        std::cout << "argv[9]   : int n_features" << std::endl;
        std::cout << "argv[10+i] : float feature_value_i" << std::endl;
        return -1;
    }

    int j = 2;
    int size = atoi(argv[j++]);
    float prob_rnd_move = atof(argv[j++]);
    int training_episodes = atoi(argv[j++]);
    int log_episodes = atoi(argv[j++]);
    int output_type = atoi(argv[j++]);
    int seed = atoi(argv[j++]);
    int world_seed = atoi(argv[j++]);
    int n_features = atoi(argv[j++]);
    if (n_features < 1 or argc != j + n_features)
    {
        std::cout << "Not enough arguments (got " << argc
                  << " expected " << j + n_features << ")" << std::endl;
        return -1;
    }

    float fv[100] = { 0.0 }; // no more than 100 features
    for (int i = 0; i < n_features; i++)
    {
        fv[i] = atof(argv[j+i]);
    }

    Random<double>* random = new Random<double>;
    random->reseed(seed);
    RLProblem<double>* problem = new GridWorld<double>(random,
                                                 world_seed,
                                                 size,
                                                 prob_rnd_move,
                                                 n_features,
                                                 fv);

    Hashing<double>* hashing = new MurmurHashing<double>(/*random*/random,
                                                   /*memorySize*/10000);
    Projector<double>* projector =
        new TileCoderHashing<double>(/*hashing*/hashing,
                                  /*nbInputs*/problem->dimension(),
                                  /*gridResolution*/10,
                                  /*nbTilings*/10,
                                  /*includeActiveFeature*/true);

    StateToStateAction<double>* toStateAction =
        new StateActionTilings<double>(/*projector*/projector,
                                    /*actions*/problem->getDiscreteActions());
    //StateToStateAction<double>* toStateAction =
    //    new TabularAction<double>(/*projector*/projector,
    //                              /*actions*/problem->getDiscreteActions());
    Trace<double>* e = new RTrace<double>(projector->dimension());

    double alpha = 0.05 / projector->vectorNorm();
    double gamma = 0.999;
    double lambda = 0.6;
    Q<double>* q = new Q<double>(alpha,
                                 gamma,
                                 lambda,
                                 e,
                                 problem->getDiscreteActions(),
                                 toStateAction);
    double epsilon = 0.05;
    Policy<double>* acting =
        new EpsilonGreedy<double>(random, problem->getDiscreteActions(), q, epsilon);
    OffPolicyControlLearner<double>* control =
        new QControl<double>(acting, toStateAction, q);

    int maxSteps = 500;
    int nbRuns = 1;
    RLAgent<double>* agent = new LearnerAgent<double>(control);
    RLRunner<double>* sim =
        new RLRunner<double>(agent, problem, maxSteps, training_episodes, nbRuns);
    sim->setVerbose(false);
    sim->runEpisodes();

    delete acting;
    delete control;
    delete agent;
    delete sim;

    // simulation with learned agent
    epsilon = 0.0;
    q->setLearn(false);
    problem->setLog(true);

    acting = new EpsilonGreedy<double>(random, problem->getDiscreteActions(), q, epsilon);
    control = new QControl<double>(acting, toStateAction, q);
    agent = new LearnerAgent<double>(control);
    sim = new RLRunner<double>(agent, problem, maxSteps, log_episodes, nbRuns);
    sim->setVerbose(false);
    sim->setEnableLog(true);
    sim->setOutputType(output_type);
    sim->runEpisodes();

    delete random;
    delete problem;
    delete hashing;
    delete projector;
    delete toStateAction;
    delete e;
    delete q;
    delete acting;
    delete control;
    delete agent;
    delete sim;
    return 0;
}



int main(int argc, char* argv[])
{
    if (argc < 2)
        return -1;
    char* name = argv[1];
    if (strcmp("mountain_car", name) == 0)
        return mountain_car(argc, argv);
    if (strcmp("grid_world", name) == 0)
        return grid_world(argc, argv);
    return -1;
}


