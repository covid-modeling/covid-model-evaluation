# Modeling Covid-19 data

Modeling Covid-19 (MC19) is an epidemiological model of COVID-19 available at [modelingcovid.com](https://modelingcovid.com). We retrieved data from the latest simulations performed by the model's maintainers, using the website's GraphQL API. The model and its GraphQL API are open-source and available on [GitHub](https://github.com/modelingcovid/covidmodel).

We used the commands below to retrieve the data for each state:

```bash
> QUERY="$(cat query_state.json | sed -e 's:":\\\":g' | tr -s '\n' ' ')"
> curl -X POST -H 'Content-Type: application/json' -d "{\"query\": \"$QUERY\"}" https://modelingcovid.com/api/graphql | jq > output_state.json
```

with the relevant queries in [_queries/](./_queries/).
