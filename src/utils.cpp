// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "utils.h"
#include <fstream>
#include <sstream>
#include "def.h"

namespace swiftware::hpp {

    //TODO add necessary includes

    // Do not change the following function signatures
    DenseMatrix *readCSV(const std::string &filename, bool removeFirstRow) {
        std::ifstream file(filename);
        std::string line, word;
        // determine number of columns in file
        std::vector<std::string> lines;
        int cntr = 0;
        while (getline(file, line)) {
            lines.push_back(line);
        }
        if (removeFirstRow) {
            lines.erase(lines.begin());
        }
        std::vector<std::vector<float>> valuesPerLine(lines.size());
        for (int i = 0; i < lines.size(); i++) {
            std::stringstream lineStream(lines[i]);
            while (getline(lineStream, word, ',')) {
                valuesPerLine[i].push_back(std::stof(word));
            }
        }
        auto *OutMat = new DenseMatrix(valuesPerLine.size(), valuesPerLine[0].size());
        auto *data = OutMat->data.data();
        int ncol = OutMat->n;
        for (int i = 0; i < valuesPerLine.size(); i++) {
            size_t cols = valuesPerLine[i].size();
            for (uint j = 0; j < cols; j++) {
                data[i * ncol + j] = valuesPerLine[i][j];
            }
        }
        return OutMat;
    }

    CSR denseToCSR(const DenseMatrix &dense)
    {
        int m = dense.m;
        int n = dense.n;

        CSR csr;
        csr.m = m;
        csr.n = n;
        csr.Ap.resize(m + 1);

        for (int i = 0; i < m; i++)
        {
            csr.Ap[i] = csr.Ax.size();

            for (int j = 0; j < n; j++)
            {
                float val = dense.data[i * n + j];
                if (val != 0.0f)
                {
                    csr.Ai.push_back(j);
                    csr.Ax.push_back(val);
                }
            }
        }

        csr.Ap[m] = csr.Ax.size();
        return csr;
    }


}