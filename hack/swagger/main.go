/*
Copyright 2024 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/common"
	builderutil "k8s.io/kube-openapi/pkg/openapiconv"
	"k8s.io/kube-openapi/pkg/validation/spec"

	trainer "github.com/kubeflow/trainer/pkg/apis/trainer/v1alpha1"
)

// Generate Kubeflow Training OpenAPI specification.
func main() {
	var oAPIDefs = map[string]common.OpenAPIDefinition{}
	defs := spec.Definitions{}

	refCallback := func(name string) spec.Ref {
		return spec.MustCreateRef("#/definitions/" + swaggify(name))
	}

	for k, v := range trainer.GetOpenAPIDefinitions(refCallback) {
		oAPIDefs[k] = v
	}

	for defName, val := range oAPIDefs {
		// Exclude InternalEvent from the OpenAPI spec since it requires runtime.Object dependency.
		if defName != "k8s.io/apimachinery/pkg/apis/meta/v1.InternalEvent" {
			// OpenAPI generator incorrectly creates models if enum doesn't have default value.
			// Therefore, we remove the default value when it is equal to ""
			// Kubernetes OpenAPI spec doesn't have enums: https://github.com/kubernetes/kubernetes/issues/109177
			for property, schema := range val.Schema.Properties {
				if schema.Enum != nil && schema.Default == "" {
					schema.Default = nil
					val.Schema.SetProperty(property, schema)
				}
			}
			defs[swaggify(defName)] = val.Schema
		}
	}
	swagger := spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger:     "2.0",
			Definitions: defs,
			Paths:       &spec.Paths{Paths: map[string]spec.PathItem{}},
			Info: &spec.Info{
				InfoProps: spec.InfoProps{
					Title:   "Kubeflow Trainer OpenAPI Spec",
					Version: "unversioned",
				},
			},
		},
	}
	swaggerOpenAPIV3 := builderutil.ConvertV2ToV3(&swagger)

	jsonBytes, err := json.MarshalIndent(swaggerOpenAPIV3, "", "  ")
	if err != nil {
		klog.Fatal(err.Error())
	}
	fmt.Println(string(jsonBytes))
}

func swaggify(name string) string {
	name = strings.Replace(name, "github.com/kubeflow/trainer/pkg/apis/", "", -1)
	name = strings.Replace(name, "sigs.k8s.io/jobset/api/", "", -1)
	name = strings.Replace(name, "k8s.io", "io.k8s", -1)
	name = strings.Replace(name, "/", ".", -1)
	return name
}
