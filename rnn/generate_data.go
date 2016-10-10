package main

import (
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
)

func main() {
	os.Remove("output.tfrecord")
	file, err := os.Create("output.tfrecord")
	if err != nil {
		log.Fatal(err)
	}

	for i := 0; i < 1; i++ {
		num_interfaces := rand.Intn(15) + 1
		distance := make([]int, num_interfaces)
		reachable := make([]int, num_interfaces)
		for j := 0; j < num_interfaces; j++ {
			reachable[j] = rand.Intn(2)
			distance[j] = rand.Intn(100)
		}
		found := -1
		min_distance := 1000
		for j := 0; j < num_interfaces; j++ {
			if reachable[j] == 1 && distance[j] < min_distance {
				found = j
				min_distance = distance[j]
			}
		}

		// We're hard coding the go protobuff API here since well fuck it.
		io.WriteString(file, "feature_lists: {\n")
		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"interface_identifier\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, fmt.Sprintf("value: %d\n", j+1))
		}
		io.WriteString(file, "value: 0\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"reachable\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, fmt.Sprintf("value: %d\n", reachable[j]))
		}
		io.WriteString(file, "value: 0\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"distance\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, fmt.Sprintf("value: %d\n", distance[j]))
		}
		io.WriteString(file, "value: 0\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"distance\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, fmt.Sprintf("value: %d\n", distance[j]))
		}
		io.WriteString(file, "value: 0\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"active_output\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, "value: 0\n")
		}
		io.WriteString(file, "value: 1\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "feature_list: {\n")
		io.WriteString(file, "key: \"selected_interface\"\n")
		io.WriteString(file, "value: {\n")
		io.WriteString(file, "feature: {\n")
		io.WriteString(file, "int64_list: {\n")
		for j := 0; j < num_interfaces; j++ {
			io.WriteString(file, "value: 0\n")
		}
		io.WriteString(file, fmt.Sprintf("value: %d\n", found+1))
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")
		io.WriteString(file, "}\n")

		io.WriteString(file, "}\n")
	}
}
