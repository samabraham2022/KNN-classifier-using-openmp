#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace std;

// A structure to hold the features of an instance
struct Instance {
    std::vector<double> features;
    int label;
};

// Calculate the distance between two instances
double euclidean_distance(const Instance& a, const Instance& b) {
    double distance = 0.0;
    for (int i = 0; i < a.features.size(); ++i) {
        distance += pow(a.features[i] - b.features[i], 2);
    }
    return sqrt(distance);
}
Instance parse_csv_line(const std::string& line) {
    Instance instance;
    std::stringstream ss(line);
    std::string field;
    int field_count = 0;
    while (std::getline(ss, field, ',')) {
        ++field_count;
        if (field_count < 4) {
            instance.features.push_back(std::stod(field));
        } else {
            instance.label = std::stoi(field);
        }
    }
    return instance;
}
std::vector<Instance> read_csv_file(const std::string& filename) {
    std::vector<Instance> dataset;
    std::ifstream file(filename);
    std::string line;
    int count = 0;
    // Skip the first line (header)
    std::getline(file, line);
    while (std::getline(file, line)) {
        dataset.push_back(parse_csv_line(line));
        count++;
    }
    cout<<"Total data read: "<<count<<"\n";
    return dataset;
}

// Classify an instance using KNN algorithm
int knn_classify(const std::vector<Instance>& dataset, const Instance& instance, int k) {
    // Calculate distances between the instance and all instances in the dataset
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < dataset.size(); ++i) {
        double distance = euclidean_distance(dataset[i], instance);
        distances.emplace_back(distance, i);
    }

    // Sort the distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Find the K nearest neighbors
    std::vector<int> neighbors;
    for (int i = 0; i < k; ++i) {
        neighbors.push_back(dataset[distances[i].second].label);
    }

    // Count the frequency of each label in the neighbors
    std::vector<int> frequency(k);
    
    for (int label : neighbors) {
        ++frequency[label];
    }

    // Find the most frequent label
    int max_frequency = 0, predicted_label;
    for (int i = 0; i < k; ++i) {
        if (frequency[i] > max_frequency) {
            max_frequency = frequency[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}

int main() {
    cout<<"Serial\n";
    std::vector<Instance> dataset = read_csv_file("training_accel.csv");
    auto start_time = std::chrono::high_resolution_clock::now();

    // Print the dataset
    for (const auto& instance : dataset) {
        for (const auto& feature : instance.features) {
        }
    }

    // Classify a test instance
    Instance test_instance = {{-0.0166,0.0001,0.0191}, 1};
    int k = 3;
    int predicted_label = knn_classify(dataset, test_instance, k);

    // Print the predicted label
    std::cout << "Predicted label: " << predicted_label << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << elapsed_time.count() << " ms" << std::endl;

    return 0;
}
