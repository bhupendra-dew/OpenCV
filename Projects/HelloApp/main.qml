// This is our previous ui
/*
import QtQuick
import QtQuick.Controls.Basic

ApplicationWindow {
    visible : true
    width : 600
    height : 500
    title : "Hello App"

    Text {
        anchors.centerIn : parent
        text : "Hello World"
        font.pixelSize : 24
    }
}
*/

// This is the improved UI

import QtQuick 
import QtQuick.Controls.Basic

ApplicationWindow {
    visible : true
    width : 400
    height : 600
    title : "Hellp App"

    property string currTime : "00 : 00 : 00"
    property QtObject backend

    Rectangle {
        anchors.fill : parent

        Image {
            sourceSize.width : parent.width
            sourceSize.height : parent.height
            source : "student.jpg"
            fillMode : Image.PreserveAspectCrop
        }
        Rectangle {
            anchors.fill : parent
            color : "transparent"

            Text {
                anchors {
                    bottom : parent.bottom
                    bottomMargin : 12
                    left : parent.left
                    leftMargin : 12
                }
                text : currTime   // used to be ;  text : "16 : 38 : 33"
                font.pixelSize : 48
                color : "black"
            }
        }
        Connections {
            target : backend

            function onUpdated(msg) {
                currTime = msg;
            }
        }
    }
}