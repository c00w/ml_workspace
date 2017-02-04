package main

import (
    "os"
    "log"
    "math/rand"

    "github.com/c00w/ml_workspace/protos/tensorflow_example"
    "github.com/golang/protobuf/proto"
)

type Decision struct {
    interface_id float32
    size float32
    value float32
    delivered_value float32
}

func generate_decisions(size int64, num_int int64) []Decision {
    output := make([]Decision, size)

    pipe_size := []int64{rand.Int63n(10000), rand.Int63n(10000)}
    pipe_used := []int64{0, 0}
    for i := int64(0); i < size; i++ {
        packet_size := rand.Int63n(10000)
        interface_id := rand.Int63n(num_int)
        value := rand.Intn(10000)
        delivered_value := 0

        if pipe_used[interface_id] +packet_size < pipe_size[interface_id] {
            delivered_value =value;
        }
        pipe_used[interface_id] += packet_size
        for j := int64(0); j < num_int; j+=1 {
            pipe_used[j] -= pipe_size[j] / 10
        }

        output[i].interface_id = float32(interface_id)
        output[i].size = float32(packet_size)
        output[i].value = float32(value)
        output[i].delivered_value = float32(delivered_value)
    }
    return output
}

func write_to_pb(rows []Decision) *tensorflow.SequenceExample {
    output := &tensorflow.SequenceExample{
        FeatureLists: &tensorflow.FeatureLists{
            FeatureList: make(map[string]*tensorflow.FeatureList),
        },
    }

    interface_ids := make([]float32, len(rows))
    for i := range rows {
        interface_ids[i] = rows[i].interface_id
    }
    output.FeatureLists.FeatureList["interface_ids"] = &tensorflow.FeatureList{
        Feature:[]*tensorflow.Feature{&tensorflow.Feature{
            Kind:
                &tensorflow.Feature_FloatList{
                    FloatList: &tensorflow.FloatList{
                        Value: interface_ids,
    }}}}}

    size := make([]float32, len(rows))
    for i := range rows {
        size[i] = rows[i].size
    }
    output.FeatureLists.FeatureList["size"] = &tensorflow.FeatureList{
        Feature:[]*tensorflow.Feature{&tensorflow.Feature{
            Kind:
                &tensorflow.Feature_FloatList{
                    FloatList: &tensorflow.FloatList{
                        Value: size,
    }}}}}

    value := make([]float32, len(rows))
    for i := range rows {
        value[i] = rows[i].value
    }
    output.FeatureLists.FeatureList["value"] = &tensorflow.FeatureList{
        Feature:[]*tensorflow.Feature{&tensorflow.Feature{
            Kind:
                &tensorflow.Feature_FloatList{
                    FloatList: &tensorflow.FloatList{
                        Value: value,
    }}}}}

    delivered_value := make([]float32, len(rows))
    for i := range rows {
        delivered_value[i] = rows[i].delivered_value
    }
    output.FeatureLists.FeatureList["delivered_value"] = &tensorflow.FeatureList{
        Feature:[]*tensorflow.Feature{&tensorflow.Feature{
            Kind:
                &tensorflow.Feature_FloatList{
                    FloatList: &tensorflow.FloatList{
                        Value: delivered_value,
    }}}}}

    return output
}

// Generates a set of records of packets sent and the returned value.
func main() {

    file, err := os.Create(os.Args[1])
    if err != nil {
        log.Fatal(err)
    }

    example_count := int64(10000)
    output := make([]*tensorflow.SequenceExample, example_count)
    for i := int64(0); i < example_count; i += 1 {
        rows := generate_decisions(100, 2)
        output[i] = write_to_pb(rows)
    }

    model := tensorflow.Model{Examples: output}
    proto.MarshalText(file, &model)
}
