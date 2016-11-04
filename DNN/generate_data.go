package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
)

func main() {
	os.Remove("output.csv")
	file, err := os.Create("output.csv")
	if err != nil {
		log.Fatal(err)
	}

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for i := 0; i < 10000; i++ {
		num_interfaces := rand.Intn(10) + 1
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
		for j := 0; j < num_interfaces; j++ {
			f := 0.0
			if j == found {
				f = 1.0
			}
			writer.Write([]string{fmt.Sprint(reachable[j]), fmt.Sprint(distance[j]), fmt.Sprint(f)})
		}
	}
}
