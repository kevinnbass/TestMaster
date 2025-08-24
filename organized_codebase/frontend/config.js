const hostName = window.location.hostname;
const isDev = hostName !== 'anvaka.github.io';
const server = isDev ? `http://${hostName}:8080/` : 'https://anvaka.github.io/map-of-github-data/';
const params = new URLSearchParams(window.location.search);
const version = params.get('v') || 'v2';

export default {
  serverUrl: '',
  // vectorTilesSource: 'http://192.168.86.79:8082/data/cities.json',
  vectorTilesTiles: `${server}${version}/points/{z}/{x}/{y}.pbf`,
  glyphsSource: `${server}/fonts/{fontstack}/{range}.pbf`,
  bordersSource: `${server}${version}/borders.geojson`,
  placesSource: `${server}${version}/places.geojson`,

  namesEndpoint: `${server}${version}/names`,
  graphsEndpoint: `${server}${version}/graphs`,
};