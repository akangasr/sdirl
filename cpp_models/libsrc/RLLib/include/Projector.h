/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Projector.h
 *
 *  Created on: Aug 18, 2012
 *      Author: sam
 */

#ifndef PROJECTOR_H_
#define PROJECTOR_H_

#include "Action.h"
#include "Tiles.h"  // Tiles
#include "Vector.h"

namespace RLLib
{

  /**
   * Feature extractor for function approximation.
   * @class T feature type
   */
  template<typename T>
  class Projector
  {
    public:
      virtual ~Projector()
      {
      }
      virtual const Vector<T>* project(const Vector<T>* x, const int& h1) =0;
      virtual const Vector<T>* project(const Vector<T>* x) =0;
      virtual T vectorNorm() const =0;
      virtual int dimension() const =0;
  };

  /**
   * @use
   * http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html
   */

  template<typename T>
  class TileCoder: public Projector<T>
  {
    protected:
      bool includeActiveFeature;
      Vector<T>* vector;
      int nbTilings;

      const Vector<T>* project(const Vector<T>* x, const int& h1, const bool& use_h1)
      {
        this->vector->clear();
        if (x->empty())
          return this->vector;
        if (this->includeActiveFeature)
        {
          this->coder(x, h1, use_h1);
          this->vector->setEntry(this->vector->dimension() - 1, 1.0);
        }
        else
          this->coder(x, h1, use_h1);
        return this->vector;
      }

    public:
      TileCoder(const int& memorySize, const int& nbTilings, bool includeActiveFeature = true) :
          includeActiveFeature(includeActiveFeature),
          vector(new SVector<T>(includeActiveFeature ? memorySize + 1 : memorySize)),
          nbTilings(nbTilings)
      {
      }

      virtual ~TileCoder()
      {
        delete this->vector;
      }

      virtual void coder(const Vector<T>* x, const int& h1, const bool& use_h1) =0;

      const Vector<T>* project(const Vector<T>* x, const int& h1)
      {
        return this->project(x, h1, true);
      }

      const Vector<T>* project(const Vector<T>* x)
      {
        return this->project(x, 0, false);
      }

      T vectorNorm() const
      {
        return this->includeActiveFeature ? this->nbTilings + 1 : this->nbTilings;
      }

      int dimension() const
      {
        return this->vector->dimension();
      }

  };

  template<typename T>
  class TileCoderHashing: public TileCoder<T>
  {
    private:
      typedef TileCoder<T> Base;  // base class
      Vector<T>* gridResolutions;  // ?
      Vector<T>* inputs;  // ?
      Tiles<T>* tiles;  // ?

    public:
      TileCoderHashing(Hashing<T>* hashing, const int& nbInputs, const T& gridResolution,
          const int& nbTilings, const bool& includeActiveFeature = true) :
          TileCoder<T>(hashing->getMemorySize(), nbTilings, includeActiveFeature),
          gridResolutions(new PVector<T>(nbInputs)),
          inputs(new PVector<T>(nbInputs)),
          tiles(new Tiles<T>(hashing))
      {
        this->gridResolutions->set(gridResolution);
      }

      TileCoderHashing(Hashing<T>* hashing, const int& nbInputs, Vector<T>* gridResolutions,
          const int& nbTilings, const bool& includeActiveFeature = true) :
          TileCoder<T>(hashing->getMemorySize(), nbTilings, includeActiveFeature),
          gridResolutions(new PVector<T>(nbInputs)),
          inputs(new PVector<T>(nbInputs)),
          tiles(new Tiles<T>(hashing))
      {
        this->gridResolutions->set(gridResolutions);
      }

      virtual ~TileCoderHashing()
      {
        delete this->inputs;
        delete this->tiles;
      }

      /**
       * Sets this->inputs to x somehow multiplied by this->gridResolutions
       * then sets this->vector to hashed values
       */
      void coder(const Vector<T>* x, const int& h1, const bool& use_h1)
      {
        this->inputs->clear();
        this->inputs->addToSelf(x)->ebeMultiplyToSelf(this->gridResolutions);
        if (use_h1 == true)
          this->tiles->tiles(/*not const*/Base::vector, Base::nbTilings, this->inputs, h1);
        else
          this->tiles->tiles(/*not const*/Base::vector, Base::nbTilings, this->inputs);
      }
  };

} // namespace RLLib

#endif /* PROJECTOR_H_ */
