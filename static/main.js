function load1(data) {
  // Load the Visualization API and the piechart package.
  google.charts.load('current', { 'packages': ['corechart'] });
  // Set a callback to run when the Google Visualization API is loaded.
  let func = drawChart.bind(drawChart, data)
  google.charts.setOnLoadCallback(func);
}
function load2(data) {
  google.load("visualization", "1", { packages: ["corechart", 'bar'] });
  let func = drawChart2.bind(drawChart2, data)
  google.setOnLoadCallback(func);
}
function drawChart(jsonData) {

  // Create the data table.
  var data = new google.visualization.DataTable();
  data.addColumn('string', 'Topping');
  data.addColumn('number', 'Slices');
  console.log(jsonData);
  jsonData.forEach((item, index) => {
    console.log(item);
    data.addRow([item.date, item.value]);
  })

  // Set chart options
  var options = {
    titleTextStyle: {
      color: '#00abf1',    // any HTML string color ('red', '#cc00cc')
      fontName: 'Robot', // i.e. 'Times New Roman'
      fontSize: 16, // 12, 18 whatever you want (don't specify px)
      bold: true,    // true or false
      italic: true  // true of false
    },
    'title': 'Figure.2: Relevance Ratio.',
    'pieHole': 0.4,
    'width': 800,
    'height': 800,
    'slices': {
      0: { color: '#E4253C' },
      1: { color: '#DEA73A' },
      2: { color: '#4C9F45' },
      3: { color: '#C5202E' },
      4: { color: '#F0412B' },
      5: { color: '#29BEE2' },
      6: { color: '#FAC315' },
      7: { color: '#A21C44' },
      8: { color: '#F26A2C' },
      9: { color: '#DD1768' },
      10: { color: '#F99D27' },
      11: { color: '#BE8B2C' },
      12: { color: '#417F45' },
      13: { color: '#1C97D3' },
      14: { color: '#5DBB47' },
      15: { color: '#06699E' },
      16: { color: '#18486B' },

    }
  };

  // Instantiate and draw our chart, passing in some options.
  var chart = new google.visualization.PieChart(document.getElementById('chart_div'));
  chart.draw(data, options);
}
function drawChart2(jsonData) {

  var data = new google.visualization.DataTable();
  data.addColumn('string', 'SDG');
  data.addColumn('number', '');
  data.addColumn({ type: 'string', role: 'style' });


  console.log(jsonData);
  data.addRows(17);
  jsonData.forEach((item, index) => {
    console.log(item);
    // data.addRow([item.date, item.value]);
    switch (index) {

      case 0:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#E4253C');
        break;
      case 1:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#DEA73A');
        break;
      case 2:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#4C9F45');
        break;
      case 3:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#C5202E');
        break;
      case 4:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#F0412B');
        break;
      case 5:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#29BEE2');
        break;
      case 6:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#FAC315');
        break;
      case 7:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#A21C44');
        break;
      case 8:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#F26A2C');
        break;
      case 9:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#DD1768');
        break;
      case 10:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#F99D27');
        break;
      case 11:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#BE8B2C');
        break;
      case 12:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#417F45');
        break;
      case 13:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#1C97D3');
        break;
      case 14:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#5DBB47');
        break;
      case 15:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#06699E');
        break;
      case 16:
        data.setValue(index, 0, item.date);
        data.setValue(index, 1, item.value);
        data.setValue(index, 2, '#18486B');
        break;
      default:
        data.addRow([item.date, item.value, 'color: #DEA73A'])
    }
  })
  var options = {
    titleTextStyle: {
      color: '#00abf1',    // any HTML string color ('red', '#cc00cc')
      fontName: 'Robot', // i.e. 'Times New Roman'
      fontSize: 16, // 12, 18 whatever you want (don't specify px)
      bold: true,    // true or false
      italic: true  // true of false
    },
    width: 1000,
    height: 600,
    title: 'Figure.1: Text Relevance.',
    //hAxis: {
    //	title: 'Rate',
    //	titleTextStyle: {
    //		color: 'red'
    //	}

    //},
    //vAxis: { format: '#%' },
    'colors': ['#FFFFFF'],
    //hAxis: {
    //  title: 'Rate',
    //  minValue: 0,
    //},
    bars: 'horizontal',
    //axes: {
    //   y: {
    //    0: { side: 'right' }
    //  }
    //}

  };

  //var chart = new google.visualization.ColumnChart(document.getElementById('chart_div2'));
  //chart.draw(data, options);
  //}
  var chart = new google.visualization.BarChart(document.getElementById('chart_div2'));
  chart.draw(data, options);
}

function updateStatus(text) {
  $('#status').text(text)
}
function updateUploadFileHint(target) {
  $("#uploadFileHint").text(target.value)
}
var socket = io();
var types = {
  CONNECT: 'connect',
  DATA: 'data',
  UPLOAD: 'upload'
}
socket.on(types.CONNECT, () => {
  updateStatus('Analyse channel is ready, please upload your file.')
  socket.on(types.UPLOAD, data => {
    updateStatus(data)
  })
  socket.on(types.DATA, data => {
    updateStatus('Analyse finished.')
    updateStatus('Please wait to draw the charts..')
    stop()
    load1(data)
    load2(data)
  })
})
var hour, minute, second;//时 分 秒
hour = minute = second = 0;//初始化
var millisecond = 0;//毫秒
var int;

function start()//开始
{
  int = setInterval(timer, 50);
}

function timer()//计时
{
  millisecond = millisecond + 50;
  if (millisecond >= 1000) {
    millisecond = 0;
    second = second + 1;
  }
  if (second >= 60) {
    second = 0;
    minute = minute + 1;
  }

  if (minute >= 60) {
    minute = 0;
    hour = hour + 1;
  }
  let costTime = second + 's ' + millisecond + 'ms';
  updateStatus(costTime)
}

function stop()//暂停
{
  window.clearInterval(int);
}
$('#analysis').on('click', () => {
  let form = $('#frm')[0]
  let form_data = new FormData(form)
  if (!form_data.get('file').name) {
    alert('Please upload your file!')
    return
  }
  let reader = new FileReader()
  let file = form_data.get('file')
  reader.onerror = () => {
    alert('Failed to upload file!')
  }
  reader.onload = event => {
    socket.emit('upload', event.target.result)
    start()
  }
  reader.readAsArrayBuffer(file)
  // $.ajax({
  //   type: 'POST',
  //   url: '/',
  //   data: form_data,
  //   contentType: false,
  //   cache: false,
  //   processData: false,
  //   success: function (data) {
  //     updateStatus(data)
  //   },
  // });
})
