#ifndef GRIDWORLD_H_
#define GRIDWORLD_H_

#include <iomanip>
#include <cassert>

#include "RL.h"
#include "Mathema.h" // Random


template<typename T>
class GridWorld: public RLLib::RLProblem<T>
{
  private:
    typedef RLLib::RLProblem<T> Base;

  protected:
    T locX;
    T locY;

    RLLib::Range<T>* locXRange;
    RLLib::Range<T>* locYRange;

    T targetX;
    T targetY;
    int size;
    float prob_rnd_move;
    int n_features;
    const float* feature_values;
    T* grid_rewards;

    struct eplog {
      int initIndex;
      std::vector<T> locXs;
      std::vector<T> locYs;
      std::vector<bool> rndMoves;
    };
    std::vector<eplog> episodes;

    void generate_world(const int& world_seed)
    {
      RLLib::Random<double>* world_gen = new RLLib::Random<double>;
      world_gen->reseed(world_seed);

      for (int i = 0; i < size * size; i++)
      {
        float reward = -0.001;
        for (int j = 0; j < n_features; j++)
        {
          if (world_gen->nextReal() < 0.3)
          {
            reward += feature_values[j] * world_gen->nextReal();
          }
        }
        if (reward > 0.0)
        {
          std::cout << "Positive weight!" << std::endl;
          for (int j = 0; j < n_features; j++)
          {
            std::cout << feature_values[j] << std::endl;
          }
          exit(-1);
        }
        this->grid_rewards[i] = (T) reward;
      }

      int index = this->targetX * this->size + this->targetY;
      this->grid_rewards[index] = (T) 1.0;

      delete world_gen;
    }

  public:
    GridWorld(RLLib::Random<T>* random = 0,
              int world_seed = 0,
              int size = 15,
              float prob_rnd_move = 0.0,
              int n_features = 1,
              const float* feature_values = 0)
        : RLLib::RLProblem<T>(/*random*/random,
                              /*nbVars*/2,
                              /*nbDiscreteActions*/4,
                              /*nbContinousActions*/0)
        , locXRange(new RLLib::Range<T>(0, size-1))
        , locYRange(new RLLib::Range<T>(0, size-1))
        , targetX(size/2)
        , targetY(size/2)
        , size(size)
        , prob_rnd_move(prob_rnd_move)
        , n_features(n_features)
        , feature_values(feature_values)
        , grid_rewards((T*)malloc(size*size*sizeof(T)))
        , episodes()
    {
      Base::discreteActions->push_back(0, 0);
      Base::discreteActions->push_back(1, 1);
      Base::discreteActions->push_back(2, 2);
      Base::discreteActions->push_back(3, 3);

      Base::observationRanges->push_back(locXRange);
      Base::observationRanges->push_back(locYRange);

      this->generate_world(world_seed);
    }

    virtual ~GridWorld()
    {
      delete this->locXRange;
      delete this->locYRange;
    }

    void updateTRStep()
    {
      Base::output->o_tp1->setEntry(0, locXRange->toUnit(locX));
      Base::output->o_tp1->setEntry(1, locYRange->toUnit(locY));

      Base::output->observation_tp1->setEntry(0, locX);
      Base::output->observation_tp1->setEntry(1, locY);
    }

    void initialize()
    {
      int initIndex = 0;
      if (Base::random)
      {
        int max = this->size - 1;
        int perimeter = max * 4;
        initIndex = Base::random->nextInt(perimeter);
        if (initIndex < max)
        {
          this->locX = initIndex;
          this->locY = 0;
        }
        else if (initIndex < max * 2)
        {
          this->locX = max;
          this->locY = initIndex - max;
        }
        else if (initIndex < max * 3)
        {
          this->locX = max * 3 - initIndex;
          this->locY = max;
        }
        else if (initIndex < max * 4)
        {
          this->locX = 0;
          this->locY = max * 4 - initIndex;
        }
        else
        {
          std::cout << "Initial location " << initIndex << " undefined" << std::endl;
          assert(false);
        }
        //std::cout << "Initial location " << initIndex << " at " << this->locX << ", " << this->locY << std::endl;
      }
      else
      {
        this->locX = 0;
        this->locY = 0;
      }
      if (this->log == true)
      {
        eplog log;
        log.initIndex = initIndex;
        log.locXs.push_back(this->locX);
        log.locYs.push_back(this->locY);
        log.rndMoves.push_back(false);
        this->episodes.push_back(log);
      }
      //std::cout << "Initial location: " << locX << ", " << locY << std::endl;
    }

    void step(const RLLib::Action<T>* a)
    {
      //std::cout << "Action: " << a->getEntry() << std::endl;
      int move = (int) a->getEntry();
      float rnd_move = Base::random->nextReal();
      bool rndMove = rnd_move < this->prob_rnd_move;
      if (rndMove)
      {
        move = Base::random->nextInt(4);
        //std::cout << "Random move" << std::endl;
      }
      switch (move)
      {
        case 0 :
          locX += 1;
          //std::cout << "Move up" << std::endl;
          break;
        case 1 :
          locX -= 1;
          //std::cout << "Move down" << std::endl;
          break;
        case 2 :
          locY += 1;
          //std::cout << "Move right" << std::endl;
          break;
        case 3 :
          locY -= 1;
          //std::cout << "Move left" << std::endl;
          break;
        default :
          std::cout << "Action " << a->getEntry() << " out of bounds" << std::endl;
          assert(false);
      }
      locX = locXRange->bound(locX);
      locY = locYRange->bound(locY);
      if (this->log == true)
      {
        this->episodes.back().locXs.push_back(this->locX);
        this->episodes.back().locYs.push_back(this->locY);
        this->episodes.back().rndMoves.push_back(rndMove);
      }
      //this->print_map();
    }

    bool endOfEpisode() const
    {
      return (locX == targetX and locY == targetY);
    }

    T r() const
    {
      int index = this->locX * this->size + this->locY;
      return this->grid_rewards[index];
    }

    T z() const
    {
      return this->r();
    }

    void draw(const int& outputType) const
    {
      if (outputType == 0)
      {
        std::cout << "(" << this->episodes.back().initIndex << ", ";
        std::cout << this->episodes.back().locXs.size() << ", ";
        std::cout << this->episodes.back().locXs[0] << ", ";
        std::cout << this->episodes.back().locYs[0] << ", ";
        std::cout << "), " << std::endl;
      }
      else if (outputType == 1)
      {
        auto i = this->episodes.back().locXs.begin();
        auto j = this->episodes.back().locYs.begin();
        auto k = this->episodes.back().rndMoves.begin();
        std::cout << "(";
        std::cout << "(" << this->episodes.back().initIndex << ", -1, -1), ";
        while (true)
        {
          std::cout << "(" << *i << ", " << *j << ", ";
          if (*k)
              std::cout << "1),";
          else
              std::cout << "0),";
          i++;
          j++;
          k++;
          if (i == this->episodes.back().locXs.end() and
              j == this->episodes.back().locYs.end())
          {
            break;
          }
        }
        std::cout << "), " << std::endl;
      }
    }

    void print_map()
    {
      //std::cout << "Location: " << this->locX << ", " << this->locY << std::endl;
      //std::cout << "Target: " << this->targetX << ", " << this->targetY << std::endl;
      std::cout << "-";
      for (int i = 0; i < this->size; i++)
      {
        std::cout << "---------";
      }
      std::cout << std::endl;
      for (int i = 0; i < this->size; i++)
      {
        std::cout << "|";
        for (int j = 0; j < this->size; j++)
        {
          int index = i * this->size + j;
          char lsym = ' ';
          char rsym = ' ';
          if (this->locX == i and this->locY == j)
          {
            lsym = '[';
            rsym = ']';
          }
          else if (this->targetX == i and this->targetY == j)
          {
            lsym = '*';
            rsym = '*';
          }
          std::cout << lsym << std::fixed << std::setfill(' ') << std::setw(6) << std::setprecision(3);
          std::cout << this->grid_rewards[index] << rsym << "|";
        }
        std::cout << std::endl;
      }
    }

};

#endif /* GRIDWORLD_H_ */
