#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[9] <= 37868.0) {
                            return 1;
                        }

                        else {
                            if (x[2] <= 69690.0) {
                                if (x[9] <= 98525.0) {
                                    return 2;
                                }

                                else {
                                    return 0;
                                }
                            }

                            else {
                                return 0;
                            }
                        }
                    }

                protected:
                };
            }
        }
    }